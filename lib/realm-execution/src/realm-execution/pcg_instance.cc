#include "realm-execution/pcg_instance.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/optimizer_attrs.h"
#include "realm-execution/dependency_set.h"
#include "realm-execution/distributed_per_device_op_state_initialization.h"
#include "realm-execution/instance_allocation.h"
#include "realm-execution/invocation_fusion.h"
#include "realm-execution/prepared_invocation.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/redops/redop_id_t.h"
#include "realm-execution/tasks/impl/fused_op_task.h"
#include "realm-execution/tasks/impl/op_task.h"
#include "realm-execution/tasks/impl/op_task_arg_register_task.h"
#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "realm-execution/tasks/impl/op_task_group_register_task.h"
#include "realm-execution/tasks/impl/zero_gradients_task.h"
#include "realm-execution/tensor_instance_backing.h"
#include "realm-execution/weight_initialization.h"
#include "task-spec/dynamic_graph/copy_insertion.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_optimizer_tensor_role.dtg.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_mapped_pcg.h"
#include "task-spec/dynamic_graph/operation_fusion.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/shard_expansion.h"
#include "task-spec/dynamic_graph/training_operation_attrs.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "utils/containers/contains.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_from_pairs.h"
#include "utils/containers/map_values.h"
#include "utils/containers/maybe_get_only.h"
#include "utils/containers/transform.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/optional.h"
#include <cstdlib>
#include <iostream>
#include <vector>

namespace FlexFlow {

PCGInstance::PCGInstance(
    RealmContext &ctx,
    std::vector<InvocationGroup> const &execution_order,
    TensorInstanceBacking const &tensor_instance_backing,
    PerDeviceOpStateBacking const &device_state_backing,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<Realm::RegionInstance> logit_grad_tensor,
    std::set<Realm::Processor> const &gradient_owning_processors)
    : ctx(ctx), execution_order(execution_order),
      tensor_instance_backing(tensor_instance_backing),
      device_state_backing(device_state_backing),
      optimizer_attrs(optimizer_attrs), logit_grad_tensor(logit_grad_tensor),
      gradient_owning_processors(gradient_owning_processors) {}

std::set<Realm::Processor> const &
    PCGInstance::get_gradient_owning_processors() const {
  return this->gradient_owning_processors;
}

PCGInstance::~PCGInstance() {
  destroy_instances(this->tensor_instance_backing,
                    ctx.get_outstanding_events());
}

RealmContext &PCGInstance::get_realm_context() {
  return this->ctx;
}

std::vector<InvocationGroup> const &PCGInstance::get_execution_order() const {
  return this->execution_order;
}

TensorInstanceBacking const &PCGInstance::get_tensor_instance_backing() const {
  return this->tensor_instance_backing;
}

PerDeviceOpStateBacking const &PCGInstance::get_device_state_backing() const {
  return this->device_state_backing;
}

OptimizerAttrs const &PCGInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}

void PCGInstance::update_optimizer_attrs_for_next_iter() {
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}

std::optional<Realm::RegionInstance>
    PCGInstance::get_loss_tensor_instance() const {
  return this->logit_grad_tensor;
}

/**
 * \brief Whether adjacent operators that one kernel can compute are folded
 * into a single operator before the passes are expanded.
 *
 * On by default; set \c FF_FUSE_OPS=0 to keep every operator separate. Not to
 * be confused with \c FF_MAX_FUSION below, which is about how many separate
 * operators are run back to back within one task.
 */
static bool operations_are_fused() {
  char const *value = std::getenv("FF_FUSE_OPS");
  return value == nullptr || std::string{value} != "0";
}

/**
 * \brief How many invocations may be fused into a single task.
 *
 * From \c FF_MAX_FUSION in the environment, where an unset variable means no
 * limit and 1 turns fusion off. Temporary, for measuring what the group size
 * is worth.
 */
static std::optional<int> get_max_fusion_group_size() {
  char const *value = std::getenv("FF_MAX_FUSION");
  if (value == nullptr) {
    return std::nullopt;
  }
  return std::stoi(value);
}

PCGInstance create_pcg_instance(
    RealmContext &ctx,
    MappedParallelComputationGraph const &mpcg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<ParallelLossConfig> const &loss,
    std::map<DynamicValueAttrs, DynamicTensorAccessor> const &input_tensors,
    DistributedFfHandle const &device_handle,
    DeviceType device_type) {

  DynamicOpenDataflowGraph dg =
      make_dynamic_open_dataflow_graph_from_mapped_pcg(mpcg, device_type);

  if (operations_are_fused()) {
    // The logit has to survive: perform_loss_insertion runs further down and
    // finds it by guid, so an operator producing it must not be folded into
    // whatever reads it.
    std::set<dynamic_tensor_guid_t> preserved_tensors;
    if (loss.has_value()) {
      preserved_tensors.insert(
          dynamic_tensor_guid_t{loss.value().logit_tensor});
    }

    // Before pass expansion, so that a fused pair gets a single backward
    // invocation and the value between them is never created in either
    // direction. Afterwards it would be too late: the backward pass would
    // already have been built around two separate operators.
    dg = perform_operation_fusion(dg, preserved_tensors);
  }

  dg = perform_pass_expansion(dg);

  std::map<DynamicValueAttrs, DynamicTensorAccessor> inputs = input_tensors;
  std::optional<DynamicValueAttrs> logit_grad_value;
  if (loss.has_value()) {
    ParallelLossConfig loss_config = assert_unwrap(loss);

    LossAttrs loss_attrs = loss_config.loss_attrs;
    GenericTensorAccessorR label_tensor = loss_config.label_tensor;
    parallel_tensor_guid_t logit_tensor = loss_config.logit_tensor;
    MappedOperatorTaskGroup loss_op_task_group = loss_config.loss_mapping;

    DynamicNodeMapping mapping = DynamicNodeMapping{
        /*op_task_group=*/loss_op_task_group,
        /*device_type=*/device_type,
    };

    auto [dg2, label_v, logit_grad_v] = perform_loss_insertion(
        dg, loss_attrs, dynamic_tensor_guid_t{logit_tensor}, mapping);
    dg = dg2;
    logit_grad_value = logit_grad_v;
    inputs.insert(std::pair{label_v, label_tensor});
  }

  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_copy_insertion(dg);
  dg = perform_shard_expansion(dg);

  TensorInstanceBacking tensor_instance_backing =
      perform_instance_allocation(dg, inputs, ctx);

  // Nothing else writes a weight before the first forward pass reads it, so
  // without this the model would train starting from whatever happened to be
  // in the memory the weights were allocated out of.
  perform_weight_initialization(
      ctx, dg, tensor_instance_backing, ctx.get_outstanding_events());

  // Optimizer state (e.g., SGD's momentum buffer) has to start at zero: the
  // update kernels accumulate into it, and the first step of SGD with momentum
  // is only equal to the plain gradient step (which is what both FlexFlow's
  // kernel and torch.optim.SGD compute) if the buffer starts out zeroed.
  for (auto const &[value, instance] : tensor_instance_backing.backing) {
    if (value.role.has_value() &&
        value.role.value().has<DynamicOptimizerTensorRole>()) {
      ctx.issue_zero_fill(/*shape=*/assert_unwrap(value.parallel_tensor_shape),
                          /*inst=*/instance.first,
                          /*requests=*/Realm::ProfilingRequestSet{},
                          /*wait_on=*/instance.second);
    }
  }

  logit_grad_value =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        for (DynamicNodeInvocation const &invocation : dg.invocations) {
          if (invocation.node_attrs.task_type != DynamicTaskType::LOSS) {
            continue;
          }
          for (auto const &[slot, value] : invocation.outputs) {
            if (slot.slot_name == TensorSlotName::LOGIT &&
                value.tensor_guid == lgv.tensor_guid &&
                value.role == lgv.role) {
              return value;
            }
          }
        }
        PANIC("couldn't find updated logit grad in the shard-expanded dynamic "
              "graph");
      });

  std::optional<Realm::RegionInstance> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return tensor_instance_backing.backing.at(lgv).first;
      });

  PerDeviceOpStateBacking device_state_backing =
      perform_distributed_per_device_op_state_initialization(
          ctx,
          dg,
          tensor_instance_backing,
          device_handle,
          optimizer_attrs,
          ctx.get_outstanding_events());

  // Compute the topological ordering of the graph
  auto [kwarg_graph, node_map] =
      labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(dg);
  std::vector<Node> node_topo_order = get_topological_ordering(kwarg_graph);
  std::vector<DynamicNodeInvocation> invocation_topo_order =
      sort_invocations_by_pass(transform(
          node_topo_order, [&](Node node) { return node_map.at_l(node); }));

  // Intern the values of the *final* graph. An id only means anything for the
  // graph it came from, so this has to happen after the last pass above, and
  // not before.
  dg = compute_value_ids_for_dynamic_open_dataflow_graph(
      compute_invocation_ids_for_dynamic_open_dataflow_graph(dg));

  std::vector<PreparedInvocation> prepared_execution_order =
      prepare_invocations(/*g=*/dg,
                          /*execution_order=*/invocation_topo_order,
                          /*tensor_instance_backing=*/tensor_instance_backing,
                          /*device_state_backing=*/device_state_backing);

  // Send each invocation's arguments to the node that will run it, once. See
  // register_op_task_args for why.
  for (PreparedInvocation const &prepared : prepared_execution_order) {
    std::optional<global_device_id_t> device_id = maybe_get_only(
        assert_unwrap(prepared.invocation.node_attrs.device_ids));
    if (!device_id.has_value()) {
      continue;
    }
    Realm::Processor target_proc =
        ctx.processor_from_global_device_id(device_id.value());

    spawn_op_task_arg_register_task(
        /*ctx=*/ctx,
        /*target_proc=*/target_proc,
        /*invocation_id=*/prepared.invocation_id,
        /*args=*/
        OpTaskArgs{
            prepared.invocation,
            prepared.tensor_backing,
            prepared.device_state,
            device_handle.at(target_proc),
            optimizer_attrs,
        },
        /*precondition=*/ctx.get_outstanding_events());
  }
  ctx.get_outstanding_events().wait();

  // TMP EXPERIMENT: structure of the gradient reduction nodes.
  if (std::getenv("FF_DUMP_REDUCTIONS") != nullptr) {
    std::map<int, int> fanin;
    size_t in_bytes = 0, out_bytes = 0;
    int nodes = 0;
    for (DynamicNodeInvocation const &invocation : dg.invocations) {
      if (!assert_unwrap(invocation.node_attrs.op_attrs)
               .has<GradientReductionAttrs>()) {
        continue;
      }
      nodes += 1;
      fanin[invocation.inputs.size()] += 1;
      for (auto const &[slot, value] : invocation.inputs) {
        in_bytes += size_t{
            get_piece_size_in_bytes(assert_unwrap(value.parallel_tensor_shape))
                .unwrap_num_bytes()
                .unwrap_nonnegative()};
      }
      for (auto const &[slot, value] : invocation.outputs) {
        out_bytes += size_t{
            get_piece_size_in_bytes(assert_unwrap(value.parallel_tensor_shape))
                .unwrap_num_bytes()
                .unwrap_nonnegative()};
      }
    }
    std::cout << "gradient reduction nodes: " << nodes << std::endl;
    for (auto const &[n, count] : fanin) {
      std::cout << "  fan-in " << n << ": " << count << " nodes" << std::endl;
    }
    std::cout << "  input bytes " << (in_bytes / 1048576) << " MiB, output "
              << (out_bytes / 1048576) << " MiB" << std::endl;
  }

  // Which gradients have to be cleared before a backward pass, and which are
  // simply written by whatever produces them. See bwd_task_overwrites_grads.
  std::map<DynamicValueAttrs, int> num_writers_by_gradient;
  std::set<DynamicValueAttrs> gradients_needing_zeroing;
  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    for (auto const &[slot, value] : invocation.outputs) {
      if (value.role != mk_dynamic_tensor_role_bwd()) {
        continue;
      }
      num_writers_by_gradient[value] += 1;
      if (!bwd_task_overwrites_grads(
              assert_unwrap(invocation.node_attrs.op_attrs))) {
        gradients_needing_zeroing.insert(value);
      }
    }
  }
  for (auto const &[value, num_writers] : num_writers_by_gradient) {
    // The backward kernels overwrite the gradient they produce, so a gradient
    // with two writers would come out as whichever of them ran last rather than
    // as the sum. Where a tensor is read more than once, the passes are
    // supposed to have given each subgradient an instance of its own and
    // inserted a gradient reduction to sum them.
    ASSERT(num_writers == 1,
           "a gradient is written by more than one invocation",
           value,
           num_writers);
  }

  // Which gradients each device zeroes at the start of a backward pass, sent
  // once rather than issued as a fill per instance per iteration. See
  // zero_gradients_for_pcg_instance.
  std::map<Realm::Processor, TensorInstanceBacking> gradients_by_proc;
  for (auto const &[value, instance] : tensor_instance_backing.backing) {
    if (value.role != mk_dynamic_tensor_role_bwd()) {
      continue;
    }
    // A gradient nothing writes still has to be cleared: something may read it,
    // and nothing else will ever put anything in it.
    if (contains_key(num_writers_by_gradient, value) &&
        !contains(gradients_needing_zeroing, value)) {
      continue;
    }
    Realm::Processor target_proc =
        ctx.processor_from_global_device_id(pt_mapping_get_device_for_coord(
            assert_unwrap(value.mapping), assert_unwrap(value.shard_coord)));
    if (!contains_key(gradients_by_proc, target_proc)) {
      gradients_by_proc.insert({target_proc, TensorInstanceBacking{{}}});
    }
    gradients_by_proc.at(target_proc).backing.insert({value, instance});
  }
  for (auto const &[target_proc, gradients] : gradients_by_proc) {
    spawn_zero_gradients_register_task(
        /*ctx=*/ctx,
        /*target_proc=*/target_proc,
        /*gradients=*/gradients,
        /*precondition=*/ctx.get_outstanding_events());
  }

  std::vector<InvocationGroup> invocation_groups = group_invocations_for_fusion(
      prepared_execution_order, optimizer_attrs, get_max_fusion_group_size());

  // A group is named to the node running it the same way an invocation is, and
  // for the same reason. Its members' arguments are already there, so this only
  // has to say which of them the group runs; it has to happen after the loop
  // above for that reason.
  for (InvocationGroup const &group : invocation_groups) {
    if (!is_fused_invocation_group(group)) {
      continue;
    }
    Realm::Processor target_proc = ctx.processor_from_global_device_id(get_only(
        assert_unwrap(group.members.front().invocation.node_attrs.device_ids)));

    spawn_op_task_group_register_task(
        /*ctx=*/ctx,
        /*target_proc=*/target_proc,
        /*group_id=*/get_group_id_for_invocation_group(group),
        /*member_ids=*/
        transform(group.members,
                  [](PreparedInvocation const &member) {
                    return member.invocation_id;
                  }),
        /*precondition=*/ctx.get_outstanding_events());
  }
  ctx.get_outstanding_events().wait();

  return PCGInstance{
      /*ctx=*/ctx,
      /*execution_order=*/invocation_groups,
      /*tensor_instance_backing=*/tensor_instance_backing,
      /*device_state_backing=*/device_state_backing,
      /*optimizer_attrs=*/optimizer_attrs,
      /*logit_grad_tensor=*/logit_grad_tensor,
      /*gradient_owning_processors=*/keys(gradients_by_proc),
  };
}

static Realm::Event
    issue_p2p_copy(RealmContext &ctx,
                   DynamicValueAttrs const &input,
                   DynamicValueAttrs const &output,
                   TensorInstanceBacking const &tensor_instance_backing,
                   Realm::Event precondition) {
  Realm::RegionInstance src_inst =
      tensor_instance_backing.backing.at(input).first;
  Realm::RegionInstance dst_inst =
      tensor_instance_backing.backing.at(output).first;
  return ctx.issue_copy(assert_unwrap(input.parallel_tensor_shape),
                        src_inst,
                        assert_unwrap(output.parallel_tensor_shape),
                        dst_inst,
                        Realm::ProfilingRequestSet{},
                        precondition);
}

static Realm::Event
    issue_p2p_reduction(RealmContext &ctx,
                        DynamicValueAttrs const &input,
                        DynamicValueAttrs const &output,
                        TensorInstanceBacking const &tensor_instance_backing,
                        redop_id_t redop_id,
                        bool is_fold,
                        bool exclusive,
                        Realm::Event precondition) {
  Realm::RegionInstance src_inst =
      tensor_instance_backing.backing.at(input).first;
  Realm::RegionInstance dst_inst =
      tensor_instance_backing.backing.at(output).first;
  return ctx.issue_reduction(assert_unwrap(input.parallel_tensor_shape),
                             src_inst,
                             assert_unwrap(output.parallel_tensor_shape),
                             dst_inst,
                             redop_id,
                             is_fold,
                             exclusive,
                             Realm::ProfilingRequestSet{},
                             precondition);
}

static Realm::Event issue_collective_broadcast(
    RealmContext &ctx,
    DynamicValueAttrs const &input,
    std::vector<DynamicValueAttrs> const &outputs,
    TensorInstanceBacking const &tensor_instance_backing,
    Realm::Event precondition) {
  // For now we just implement this as the naive set of N p2p copies.
  std::vector<Realm::Event> result =
      transform(outputs, [&](DynamicValueAttrs const &output) {
        return issue_p2p_copy(
            ctx, input, output, tensor_instance_backing, precondition);
      });
  return Realm::Event::merge_events(result);
}

static Realm::Event issue_collective_reduction(
    RealmContext &ctx,
    std::vector<DynamicValueAttrs> const &inputs,
    DynamicValueAttrs const &output,
    TensorInstanceBacking const &tensor_instance_backing,
    redop_id_t redop_id,
    Realm::Event precondition) {
  // For now we just implement this as a naive set of N p2p reductions. Because
  // we're launching them in parallel they cannot be exclusive (i.e., they need
  // to use per-element atomics to update the output tensor)
  std::vector<Realm::Event> result =
      transform(inputs, [&](DynamicValueAttrs const &input) {
        return issue_p2p_reduction(ctx,
                                   input,
                                   output,
                                   tensor_instance_backing,
                                   redop_id,
                                   /*is_fold*/ false,
                                   /*exclusive*/ false,
                                   precondition);
      });
  return Realm::Event::merge_events(result);
}

/**
 * \brief Spawn the Realm operations (tasks, copies, etc.) for a given \ref
 * DynamicNodeInvocation, given the specified dependencies, instances, etc. Note
 * that one \ref DynamicNodeInvocation may become multiple Realm operations
 * (e.g., a parallel operator may turn into multiple copies).
 */
static Realm::Event
    spawn_dynamic_node_invocation(RealmContext &ctx,
                                  PreparedInvocation const &prepared,
                                  Realm::Event precondition,
                                  OptimizerAttrs const &optimizer_attrs,
                                  DistributedFfHandle const &device_handle) {
  DynamicNodeInvocation const &invocation = prepared.invocation;
  TensorInstanceBacking const &tensor_backing = prepared.tensor_backing;
  TensorInstanceBacking const &tensor_instance_backing =
      prepared.tensor_backing;

  auto spawn_task = [&]() {
    Realm::Processor target_proc = ctx.processor_from_global_device_id(
        get_only(assert_unwrap(invocation.node_attrs.device_ids)));
    return spawn_op_task(ctx,
                         target_proc,
                         invocation,
                         prepared.invocation_id,
                         optimizer_attrs,
                         precondition);
  };

  auto issue_copy = [&]() {
    DynamicValueAttrs const &input = get_only(invocation.inputs).second;
    DynamicValueAttrs const &output = get_only(invocation.outputs).second;
    return issue_p2p_copy(
        ctx, input, output, tensor_instance_backing, precondition);
  };

  auto issue_replicate = [&]() {
    DynamicValueAttrs const &input = get_only(invocation.inputs).second;
    std::vector<DynamicValueAttrs> outputs =
        vector_of(values(invocation.outputs));
    return issue_collective_broadcast(
        ctx, input, outputs, tensor_instance_backing, precondition);
  };

  auto issue_reduction = [&]() {
    std::vector<DynamicValueAttrs> inputs =
        vector_of(values(invocation.inputs));
    DynamicValueAttrs const &output = get_only(invocation.outputs).second;
    redop_id_t redop_id = get_sum_redop_id_for_data_type(
        assert_unwrap(output.parallel_tensor_shape).data_type);
    return issue_collective_reduction(
        ctx, inputs, output, tensor_instance_backing, redop_id, precondition);
  };

  TrainingOperationAttrs op_attrs =
      assert_unwrap(invocation.node_attrs.op_attrs);
  return op_attrs.visit<Realm::Event>(overload{
      [&](PCGOperatorAttrs const &pcg_op_attrs) {
        return pcg_op_attrs.visit<Realm::Event>(overload{
            [&](InputAttrs const &) { return Realm::Event::NO_EVENT; },
            [&](WeightAttrs const &) {
              // A weight has no forward or backward task of its own, but
              // update insertion hangs the optimizer's update task off of the
              // weight's node (see perform_update_insertion), so an UPD
              // invocation for a weight does have a task to run.
              DynamicTaskType task_type =
                  assert_unwrap(invocation.node_attrs.task_type);
              if (task_type == DynamicTaskType::UPD) {
                return spawn_task();
              }
              return Realm::Event::NO_EVENT;
            },
            [&](ReplicateAttrs const &) {
              DynamicTaskType task_type =
                  assert_unwrap(invocation.node_attrs.task_type);
              switch (task_type) {
                case DynamicTaskType::FWD:
                  return issue_replicate();
                case DynamicTaskType::BWD:
                  return issue_reduction();
                default:
                  PANIC("Unhandled replicate task type ", task_type);
              }
            },
            [&](auto const &) { return spawn_task(); },
        });
      },
      [&](LossAttrs const &) { return spawn_task(); },
      [&](CopyAttrs const &) { return issue_copy(); },
      // A gradient reduction is a task like any other, so that it can be
      // fused with the backward tasks around it. See
      // group_invocations_for_fusion.
      [&](GradientReductionAttrs const &) { return spawn_task(); },
  });
}

/**
 * \brief Spawn the Realm operation for a single \ref InvocationGroup.
 *
 * A group of one is issued exactly as it was before there was any such thing
 * as a group; a larger one becomes a single fused task.
 */
static Realm::Event
    spawn_invocation_group(RealmContext &ctx,
                           InvocationGroup const &group,
                           Realm::Event precondition,
                           OptimizerAttrs const &optimizer_attrs,
                           DistributedFfHandle const &device_handle) {
  if (!is_fused_invocation_group(group)) {
    return spawn_dynamic_node_invocation(ctx,
                                         get_only(group.members),
                                         precondition,
                                         optimizer_attrs,
                                         device_handle);
  }

  Realm::Processor target_proc = ctx.processor_from_global_device_id(get_only(
      assert_unwrap(group.members.front().invocation.node_attrs.device_ids)));
  return spawn_fused_op_task(ctx,
                             target_proc,
                             get_group_id_for_invocation_group(group),
                             optimizer_attrs,
                             precondition);
}

static std::map<dynamic_layer_guid_t, Realm::Event>
    execute_distributed_dynamic_node_invocation_set(
        RealmContext &ctx,
        std::vector<InvocationGroup> const &groups,
        OptimizerAttrs const &optimizer_attrs,
        DistributedFfHandle const &device_handle) {
  // For simplicity we'll track a dependency on all outstanding operations up to
  // this point. This will create an effective barrier between phases.
  DependencySet dependency_set{ctx.get_outstanding_events()};

  std::vector<std::pair<dynamic_layer_guid_t, Realm::Event>> results;
  for (InvocationGroup const &group : groups) {
    // A group depends on everything any of its members reads or writes.
    // Dependencies between members are carried by the order the fused body
    // runs them in, so they need no events of their own.
    std::vector<Realm::Event> input_dependencies =
        transform(group.input_ids, [&](dynamic_value_id_t const &value) {
          return dependency_set.get_dependency_for_reader(value);
        });
    std::vector<Realm::Event> output_dependencies =
        transform(group.output_ids, [&](dynamic_value_id_t const &value) {
          return dependency_set.get_dependency_for_writer(value);
        });
    Realm::Event precondition = Realm::Event::merge_events(
        Realm::Event::merge_events(input_dependencies),
        Realm::Event::merge_events(output_dependencies));

    Realm::Event result = spawn_invocation_group(
        ctx, group, precondition, optimizer_attrs, device_handle);

    for (dynamic_value_id_t const &value : group.input_ids) {
      dependency_set.add_reader(value, result);
    }
    for (dynamic_value_id_t const &value : group.output_ids) {
      dependency_set.add_writer(value, result);
    }
    for (PreparedInvocation const &member : group.members) {
      results.push_back(
          std::pair{member.invocation.node_attrs.layer_guid, result});
    }
  }
  return map_from_pairs(results);
}

/**
 * \brief Zero every gradient instance held by \p pcg_instance.
 *
 * The backward kernels accumulate into their gradient tensors rather than
 * overwriting them (that is how the subgradients of a tensor that is consumed
 * more than once get summed together), so the gradients have to be cleared
 * before each backward pass or they would keep accumulating across training
 * iterations.
 *
 * One task per device does all of that device's gradients, rather than a Realm
 * fill per instance. There are more gradient instances than the backward pass
 * has tasks, so issuing them separately made zeroing the largest source of
 * Realm operations in an iteration, and Realm's fill is a strided kernel where
 * a dense instance only needs a memset.
 *
 * \note The tasks are issued through \ref RealmContext, so they end up in the
 * context's outstanding events and are therefore picked up as dependencies by
 * the \ref DependencySet that \ref
 * execute_distributed_dynamic_node_invocation_set starts from. No additional
 * synchronization is needed.
 */
static void zero_gradients_for_pcg_instance(PCGInstance &pcg_instance) {
  RealmContext &ctx = pcg_instance.get_realm_context();

  // The zeroing has to wait on everything already in flight. The barrier that
  // \ref execute_distributed_dynamic_node_invocation_set sets up only makes
  // the tasks it spawns depend on the zeroing; it does nothing to keep the
  // zeroing from running ahead of what came before it. On every iteration after
  // the first, what came before it is the previous iteration's backward and
  // update tasks, still reading and writing these very instances.
  Realm::Event precondition = ctx.get_outstanding_events();

  for (Realm::Processor const &target_proc :
       pcg_instance.get_gradient_owning_processors()) {
    spawn_zero_gradients_task(ctx, target_proc, precondition);
  }
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_all_passes_for_pcg_instance(
        PCGInstance &pcg_instance, DistributedFfHandle const &device_handle) {
  zero_gradients_for_pcg_instance(pcg_instance);

  std::vector<InvocationGroup> execution_order =
      pcg_instance.get_execution_order();
  std::map<dynamic_layer_guid_t, Realm::Event> result =
      execute_distributed_dynamic_node_invocation_set(
          /*ctx=*/pcg_instance.get_realm_context(),
          /*invocations=*/execution_order,
          /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
          /*device_handle=*/device_handle);
  pcg_instance.update_optimizer_attrs_for_next_iter();
  return result;
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_forward_pass_for_pcg_instance(
        PCGInstance &pcg_instance, DistributedFfHandle const &device_handle) {
  std::vector<InvocationGroup> execution_order = filter(
      pcg_instance.get_execution_order(), [](InvocationGroup const &group) {
        return get_task_type_for_invocation_group(group) ==
               DynamicTaskType::FWD;
      });

  return execute_distributed_dynamic_node_invocation_set(
      /*ctx=*/pcg_instance.get_realm_context(),
      /*invocations=*/execution_order,
      /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
      /*device_handle=*/device_handle);
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_backward_pass_for_pcg_instance(
        PCGInstance &pcg_instance, DistributedFfHandle const &device_handle) {
  zero_gradients_for_pcg_instance(pcg_instance);

  std::vector<InvocationGroup> execution_order = filter(
      pcg_instance.get_execution_order(), [](InvocationGroup const &group) {
        return get_task_type_for_invocation_group(group) ==
               DynamicTaskType::BWD;
      });

  return execute_distributed_dynamic_node_invocation_set(
      /*ctx=*/pcg_instance.get_realm_context(),
      /*invocations=*/execution_order,
      /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
      /*device_handle=*/device_handle);
}

std::map<dynamic_layer_guid_t, Realm::Event>
    perform_update_pass_for_pcg_instance(
        PCGInstance &pcg_instance, DistributedFfHandle const &device_handle) {
  std::vector<InvocationGroup> execution_order = filter(
      pcg_instance.get_execution_order(), [](InvocationGroup const &group) {
        return get_task_type_for_invocation_group(group) ==
               DynamicTaskType::UPD;
      });

  std::map<dynamic_layer_guid_t, Realm::Event> result =
      execute_distributed_dynamic_node_invocation_set(
          /*ctx=*/pcg_instance.get_realm_context(),
          /*invocations=*/execution_order,
          /*optimizer_attrs=*/pcg_instance.get_optimizer_attrs(),
          /*device_handle=*/device_handle);
  pcg_instance.update_optimizer_attrs_for_next_iter();
  return result;
}

} // namespace FlexFlow
