#include "task-spec/dynamic_graph/operation_fusion.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/are_disjoint.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/contains.h"
#include "utils/containers/filter_keys.h"
#include "utils/containers/filter_values.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_disjoint_maps.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/containers/set_union.h"
#include "utils/containers/transform.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<Activation>
    try_get_fusable_activation(PCGOperatorAttrs const &attrs) {
  if (!attrs.has<ElementUnaryAttrs>()) {
    return std::nullopt;
  }

  ElementUnaryAttrs element_unary_attrs = attrs.get<ElementUnaryAttrs>();

  // An Activation names the function but carries no parameters, so an
  // elementwise operator that was given one cannot be written as an activation
  // on its neighbor. SiLU's scalar is its beta, which defaults to 1.
  auto without_parameters =
      [&](Activation activation,
          float default_scalar) -> std::optional<Activation> {
    if (element_unary_attrs.scalar.value_or(default_scalar) != default_scalar) {
      return std::nullopt;
    }
    return activation;
  };

  switch (element_unary_attrs.op_type) {
    case OperatorType::RELU:
      return Activation::RELU;
    case OperatorType::SIGMOID:
      return Activation::SIGMOID;
    case OperatorType::TANH:
      return Activation::TANH;
    case OperatorType::GELU:
      return Activation::GELU;
    case OperatorType::SILU:
      return without_parameters(Activation::SILU, 1.0f);
    default:
      return std::nullopt;
  }
}

std::optional<TrainingOperationAttrs>
    try_fuse_operation_attrs(TrainingOperationAttrs const &producer,
                             TrainingOperationAttrs const &consumer) {
  if (!producer.has<PCGOperatorAttrs>() || !consumer.has<PCGOperatorAttrs>()) {
    return std::nullopt;
  }

  PCGOperatorAttrs producer_attrs = producer.require_pcg_op();

  std::optional<Activation> activation =
      try_get_fusable_activation(consumer.require_pcg_op());
  if (!activation.has_value()) {
    return std::nullopt;
  }

  if (producer_attrs.has<BatchNormAttrs>()) {
    BatchNormAttrs batch_norm_attrs = producer_attrs.get<BatchNormAttrs>();

    // Already carrying an activation, either from the model author or from an
    // earlier run of this pass. Its kernel applies one activation, not two.
    if (batch_norm_attrs.activation.has_value()) {
      return std::nullopt;
    }

    if (!batch_norm_supports_fused_activation(activation.value()) ||
        !batch_norm_mode_supports_fused_activation(batch_norm_attrs.mode)) {
      return std::nullopt;
    }

    batch_norm_attrs.activation = activation;
    return TrainingOperationAttrs{PCGOperatorAttrs{batch_norm_attrs}};
  }

  return std::nullopt;
}

/**
 * @brief The slots at which \p invocation reads \p value.
 */
static std::set<DynamicTensorSlot>
    slots_reading_value(DynamicNodeInvocation const &invocation,
                        DynamicValueAttrs const &value) {
  return keys(filter_values(invocation.inputs, [&](DynamicValueAttrs const &v) {
    return v == value;
  }));
}

bool invocations_are_structurally_fusible(DynamicNodeInvocation const &producer,
                                          DynamicNodeInvocation const &consumer,
                                          DynamicValueAttrs const &between,
                                          bool between_has_other_readers) {
  if (producer == consumer) {
    return false;
  }

  // Folding the producer away leaves nowhere to put anything else it produced.
  if (set_of(values(producer.outputs)) !=
      std::set<DynamicValueAttrs>{between}) {
    return false;
  }

  // Anything else still reading the value needs the value to exist.
  if (between_has_other_readers) {
    return false;
  }

  // A consumer that reads the value at two slots would need the fused operator
  // to produce it, which is exactly what fusing avoids.
  if (slots_reading_value(consumer, between).size() != 1) {
    return false;
  }

  // One invocation runs as one task, so the two have to be the same task's
  // worth of work: same devices, same pass.
  if (producer.node_attrs.device_ids != consumer.node_attrs.device_ids) {
    return false;
  }
  if (producer.node_attrs.task_type != consumer.node_attrs.task_type) {
    return false;
  }
  if (producer.node_attrs.mapping.has_value() !=
      consumer.node_attrs.mapping.has_value()) {
    return false;
  }
  if (producer.node_attrs.mapping.has_value()) {
    // One task cannot be in two places, so the fused operator only exists if
    // both halves were already going to run on the same machine coordinates.
    if (keys(producer.node_attrs.mapping.value()
                 .op_task_group.get_shard_bindings()
                 .as_map()) != keys(consumer.node_attrs.mapping.value()
                                        .op_task_group.get_shard_bindings()
                                        .as_map())) {
      return false;
    }
    if (producer.node_attrs.mapping.value().device_type !=
        consumer.node_attrs.mapping.value().device_type) {
      return false;
    }
  }

  // The fused invocation reads under both operators' slot names, so those names
  // have to distinguish the tensors they name.
  std::set<TensorSlotName> producer_input_slots =
      transform(keys(producer.inputs),
                [](DynamicTensorSlot const &s) { return s.slot_name; });
  std::set<TensorSlotName> retained_consumer_input_slots =
      transform(keys(filter_values(
                    consumer.inputs,
                    [&](DynamicValueAttrs const &v) { return v != between; })),
                [](DynamicTensorSlot const &s) { return s.slot_name; });
  std::set<TensorSlotName> consumer_output_slots =
      transform(keys(consumer.outputs),
                [](DynamicTensorSlot const &s) { return s.slot_name; });

  return are_disjoint(producer_input_slots, retained_consumer_input_slots) &&
         are_disjoint(producer_input_slots, consumer_output_slots) &&
         are_disjoint(retained_consumer_input_slots, consumer_output_slots);
}

/**
 * @brief The mapping for an invocation that reads under \p producer's input
 * slots and writes under \p consumer's output slots.
 *
 * @details Each operator's mapping binds its own slot names to tensor
 * coordinates, so the fused mapping takes each slot's binding from whichever of
 * the two operators that slot came from.
 */
static std::optional<DynamicNodeMapping>
    fuse_mappings(DynamicNodeInvocation const &producer,
                  DynamicNodeInvocation const &consumer,
                  DynamicNodeInvocation const &fused) {
  if (!producer.node_attrs.mapping.has_value()) {
    return std::nullopt;
  }

  DynamicNodeMapping producer_mapping = producer.node_attrs.mapping.value();
  DynamicNodeMapping consumer_mapping = consumer.node_attrs.mapping.value();

  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
      &producer_bindings = producer_mapping.op_task_group.get_shard_bindings();
  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> const
      &consumer_bindings = consumer_mapping.op_task_group.get_shard_bindings();

  std::set<TensorSlotName> producer_slots =
      transform(keys(producer.inputs),
                [](DynamicTensorSlot const &s) { return s.slot_name; });
  std::set<TensorSlotName> consumer_slots = set_union(
      transform(keys(fused.outputs),
                [](DynamicTensorSlot const &s) { return s.slot_name; }),
      set_minus(
          transform(keys(fused.inputs),
                    [](DynamicTensorSlot const &s) { return s.slot_name; }),
          producer_slots));

  bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding> fused_bindings;
  for (auto const &[machine_coord, producer_binding] : producer_bindings) {
    OperatorAtomicTaskShardBinding consumer_binding =
        consumer_bindings.at_l(machine_coord);

    fused_bindings.equate(
        machine_coord,
        OperatorAtomicTaskShardBinding{
            binary_merge_disjoint_maps(
                restrict_keys(producer_binding.tensor_coords, producer_slots),
                restrict_keys(consumer_binding.tensor_coords, consumer_slots)),
        });
  }

  return DynamicNodeMapping{
      /*op_task_group=*/MappedOperatorTaskGroup{fused_bindings},
      /*device_type=*/producer_mapping.device_type,
  };
}

std::optional<DynamicNodeInvocation>
    try_fuse_invocations(DynamicNodeInvocation const &producer,
                         DynamicNodeInvocation const &consumer,
                         DynamicValueAttrs const &between,
                         bool between_has_other_readers) {
  if (!invocations_are_structurally_fusible(
          producer, consumer, between, between_has_other_readers)) {
    return std::nullopt;
  }

  std::optional<TrainingOperationAttrs> fused_op_attrs =
      try_fuse_operation_attrs(producer.node_attrs.op_attrs.value(),
                               consumer.node_attrs.op_attrs.value());
  if (!fused_op_attrs.has_value()) {
    return std::nullopt;
  }

  std::map<DynamicTensorSlot, DynamicValueAttrs> fused_inputs =
      binary_merge_disjoint_maps(
          producer.inputs,
          filter_values(consumer.inputs, [&](DynamicValueAttrs const &v) {
            return v != between;
          }));

  DynamicNodeInvocation result = DynamicNodeInvocation{
      /*inputs=*/fused_inputs,
      /*node_attrs=*/
      DynamicNodeAttrs{
          /*task_type=*/producer.node_attrs.task_type,
          /*device_ids=*/producer.node_attrs.device_ids,
          /*mapping=*/std::nullopt,
          /*op_attrs=*/fused_op_attrs.value(),
          // The fused operator is the producer with extra work folded onto its
          // end, so it keeps the producer's identity. The consumer's layer
          // disappears along with the value between them.
          /*layer_guid=*/producer.node_attrs.layer_guid,
          /*per_device_op_state=*/producer.node_attrs.per_device_op_state,
      },
      /*outputs=*/consumer.outputs,
  };

  result.node_attrs.mapping = fuse_mappings(producer, consumer, result);

  return result;
}

/**
 * @brief Fuse as many disjoint pairs as can be found in a single traversal, or
 * \c std::nullopt if there are none.
 *
 * @details Each invocation takes part in at most one fusion per round, since
 * fusing a pair changes both of its members. Chains longer than two therefore
 * take more than one round, which is what \ref perform_operation_fusion's loop
 * is for.
 */
static std::optional<DynamicOpenDataflowGraph>
    perform_one_round_of_operation_fusion(
        DynamicOpenDataflowGraph const &g,
        std::set<dynamic_tensor_guid_t> const &preserved_tensors) {

  std::map<dynamic_value_id_t, std::set<InternalDynamicSlotSite>>
      sinks_by_value = dynamic_graph_get_sinks_of_each_value(g);

  std::set<dynamic_invocation_id_t> already_fused;
  std::set<DynamicNodeInvocation> result;
  bool fused_anything = false;

  for (DynamicNodeInvocation const &producer : get_dynamic_invocation_set(g)) {
    dynamic_invocation_id_t producer_id =
        dynamic_graph_get_id_for_invocation(g, producer);

    if (contains(already_fused, producer_id)) {
      continue;
    }

    std::optional<DynamicNodeInvocation> fused = std::nullopt;
    dynamic_invocation_id_t consumer_id = producer_id;

    if (producer.outputs.size() == 1) {
      DynamicValueAttrs between = get_only(values(producer.outputs));

      std::set<InternalDynamicSlotSite> sinks =
          try_at(sinks_by_value, dynamic_graph_get_id_for_value(g, between))
              .value_or(std::set<InternalDynamicSlotSite>{});

      // A value that something outside the graph will ask for by name has to
      // stay in the graph, whatever the traffic it costs.
      bool is_preserved = contains(preserved_tensors, between.tensor_guid);

      if (sinks.size() == 1 && !is_preserved) {
        InternalDynamicSlotSite sink = get_only(sinks);
        if (!contains(already_fused, sink.invocation_id)) {
          consumer_id = sink.invocation_id;
          fused = try_fuse_invocations(
              producer,
              dynamic_graph_get_invocation_for_id(g, consumer_id),
              between,
              /*between_has_other_readers=*/false);
        }
      }
    }

    if (fused.has_value()) {
      already_fused.insert(producer_id);
      already_fused.insert(consumer_id);
      result.insert(fused.value());
      fused_anything = true;
    }
  }

  if (!fused_anything) {
    return std::nullopt;
  }

  // Everything that was not fused this round carries over unchanged. This is
  // done in a second traversal because whether an invocation survives is only
  // settled once every pair has been considered.
  for (DynamicNodeInvocation const &invocation :
       get_dynamic_invocation_set(g)) {
    if (!contains(already_fused,
                  dynamic_graph_get_id_for_invocation(g, invocation))) {
      result.insert(invocation);
    }
  }

  return dynamic_open_dataflow_graph_from_invocation_set(result);
}

DynamicOpenDataflowGraph perform_operation_fusion(
    DynamicOpenDataflowGraph const &g,
    std::set<dynamic_tensor_guid_t> const &preserved_tensors) {

  DynamicOpenDataflowGraph result = g;
  while (true) {
    std::optional<DynamicOpenDataflowGraph> next =
        perform_one_round_of_operation_fusion(result, preserved_tensors);
    if (!next.has_value()) {
      return result;
    }
    result = next.value();
  }
}

} // namespace FlexFlow
