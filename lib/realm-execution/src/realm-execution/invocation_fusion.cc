#include "realm-execution/invocation_fusion.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/dynamic_graph/dynamic_task_type.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "utils/containers/get_only.h"
#include "utils/containers/maybe_get_only.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/optional.h"
#include "utils/overload.h"
#include <algorithm>
#include <set>

namespace FlexFlow {

/**
 * @brief The pass \p invocation belongs to.
 *
 * Copy insertion runs after the passes have been expanded, and a copy is not
 * part of any pass's task, so it is left without a task type of its own. It
 * still belongs to a pass, though -- the one that produces and consumes the
 * tensor it moves -- and that is what decides where it sorts and what it may
 * be grouped with.
 */
static DynamicTaskType
    get_pass_for_invocation(DynamicNodeInvocation const &invocation) {
  if (invocation.node_attrs.task_type.has_value()) {
    return invocation.node_attrs.task_type.value();
  }

  ASSERT(assert_unwrap(invocation.node_attrs.op_attrs).has<CopyAttrs>(),
         "a copy is the only invocation without a task type",
         invocation.node_attrs);

  return dynamic_task_type_from_tensor_role_for_copy(
      assert_unwrap(get_only(values(invocation.inputs)).role));
}

/**
 * @brief The device a fused body would run \p invocation on, or \c std::nullopt
 * if the invocation cannot be part of a fused group at all.
 *
 * This has to agree with \ref spawn_dynamic_node_invocation about which
 * invocations become a plain operator task: an invocation that turns into
 * copies, or into nothing at all, has no task body for a fused task to run.
 */
static std::optional<global_device_id_t> get_fusable_device_for_invocation(
    DynamicNodeInvocation const &invocation,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {

  DynamicNodeAttrs const &node_attrs = invocation.node_attrs;
  DynamicTaskType task_type = get_pass_for_invocation(invocation);

  bool becomes_op_task =
      assert_unwrap(node_attrs.op_attrs)
          .visit<bool>(overload{
              [&](PCGOperatorAttrs const &pcg_op_attrs) {
                // A parallel operator has no task implementation to call.
                if (!compgraph_op_attrs_from_pcg_op_attrs(pcg_op_attrs)
                         .has_value()) {
                  return false;
                }
                return pcg_op_attrs.visit<bool>(overload{
                    [](InputAttrs const &) { return false; },
                    [&](WeightAttrs const &) {
                      // Update insertion hangs the optimizer's task off of the
                      // weight's node, so a weight does have a task, but only
                      // for the update pass.
                      return task_type == DynamicTaskType::UPD;
                    },
                    [](auto const &) { return true; },
                });
              },
              [](LossAttrs const &) { return true; },
              [](CopyAttrs const &) { return false; },
              [](GradientReductionAttrs const &) { return true; },
          });

  if (!becomes_op_task) {
    return std::nullopt;
  }

  if (!get_task_id_for_op(node_attrs, optimizer_attrs).has_value()) {
    return std::nullopt;
  }

  return maybe_get_only(assert_unwrap(node_attrs.device_ids));
}

/**
 * @brief Where \p task_type falls in the order the passes run in.
 */
static int get_pass_index_for_task_type(DynamicTaskType task_type) {
  switch (task_type) {
    case DynamicTaskType::FWD:
      return 0;
    case DynamicTaskType::LOSS:
      return 1;
    case DynamicTaskType::BWD:
      return 2;
    case DynamicTaskType::UPD:
      return 3;
    default:
      PANIC("Unhandled DynamicTaskType", task_type);
  }
}

std::vector<DynamicNodeInvocation> sort_invocations_by_pass(
    std::vector<DynamicNodeInvocation> const &topological_order) {
  std::vector<DynamicNodeInvocation> result = topological_order;
  std::stable_sort(
      result.begin(),
      result.end(),
      [](DynamicNodeInvocation const &l, DynamicNodeInvocation const &r) {
        return get_pass_index_for_task_type(get_pass_for_invocation(l)) <
               get_pass_index_for_task_type(get_pass_for_invocation(r));
      });
  return result;
}

/**
 * @brief Whether issuing \p invocation produces no Realm operation at all.
 *
 * Inputs and weights have no forward or backward task of their own. Issuing one
 * yields \c Realm::Event::NO_EVENT, and putting that into a \ref
 * DependencySet leaves it exactly as it was, so such an invocation can be
 * issued at any point without changing what anything waits on. That is what
 * lets a group be built across one.
 *
 * This has to agree with \ref spawn_dynamic_node_invocation.
 */
static bool
    produces_no_realm_operation(DynamicNodeInvocation const &invocation) {
  DynamicTaskType task_type = get_pass_for_invocation(invocation);
  return assert_unwrap(invocation.node_attrs.op_attrs)
      .visit<bool>(overload{
          [&](PCGOperatorAttrs const &pcg_op_attrs) {
            return pcg_op_attrs.visit<bool>(overload{
                [](InputAttrs const &) { return true; },
                [&](WeightAttrs const &) {
                  return task_type != DynamicTaskType::UPD;
                },
                [](auto const &) { return false; },
            });
          },
          [](auto const &) { return false; },
      });
}

dynamic_invocation_id_t
    get_group_id_for_invocation_group(InvocationGroup const &group) {
  ASSERT(!group.members.empty(), "an invocation group is never empty");
  return group.members.front().invocation_id;
}

bool is_fused_invocation_group(InvocationGroup const &group) {
  return group.members.size() > 1;
}

DynamicTaskType
    get_task_type_for_invocation_group(InvocationGroup const &group) {
  ASSERT(!group.members.empty(), "an invocation group is never empty");
  return get_pass_for_invocation(group.members.front().invocation);
}

/**
 * @brief Fill in the fields of \p group that are derived from its members.
 */
static InvocationGroup complete_invocation_group(InvocationGroup group) {
  std::set<dynamic_value_id_t> input_ids;
  std::set<dynamic_value_id_t> output_ids;
  for (PreparedInvocation const &member : group.members) {
    input_ids.insert(member.input_ids.begin(), member.input_ids.end());
    output_ids.insert(member.output_ids.begin(), member.output_ids.end());
  }
  group.input_ids = vector_of(input_ids);
  group.output_ids = vector_of(output_ids);
  return group;
}

std::vector<InvocationGroup> group_invocations_for_fusion(
    std::vector<PreparedInvocation> const &execution_order,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    std::optional<int> max_group_size) {

  std::vector<InvocationGroup> result;
  InvocationGroup current;
  std::optional<global_device_id_t> current_device_id;
  std::optional<DynamicTaskType> current_task_type;

  auto flush = [&]() {
    if (!current.members.empty()) {
      result.push_back(complete_invocation_group(current));
    }
    current = InvocationGroup{};
    current_device_id = std::nullopt;
    current_task_type = std::nullopt;
  };

  for (PreparedInvocation const &prepared : execution_order) {
    std::optional<global_device_id_t> device_id =
        get_fusable_device_for_invocation(prepared.invocation, optimizer_attrs);
    DynamicTaskType task_type = get_pass_for_invocation(prepared.invocation);

    // Anything that is not issued as a plain operator task is a group of its
    // own, and ends whatever run was being built. Ending the run is what keeps
    // a group to a run of the topological order, and so keeps fusing it from
    // introducing a cycle. An invocation that produces no Realm operation at
    // all is the exception: it constrains nothing, so a run can be built right
    // across it.
    if (!device_id.has_value()) {
      if (!produces_no_realm_operation(prepared.invocation)) {
        flush();
      }
      result.push_back(
          complete_invocation_group(InvocationGroup{{prepared}, {}, {}}));
      continue;
    }

    bool joins_current =
        !current.members.empty() && current_device_id == device_id &&
        current_task_type == task_type &&
        (!max_group_size.has_value() ||
         current.members.size() < static_cast<size_t>(max_group_size.value()));
    if (!joins_current) {
      flush();
      current_device_id = device_id;
      current_task_type = task_type;
    }
    current.members.push_back(prepared);
  }
  flush();

  return result;
}

} // namespace FlexFlow
