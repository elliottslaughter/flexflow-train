#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.h"
#include "realm-execution/tasks/serializer/serializable_realm_event.h"
#include "realm-execution/tasks/serializer/serializable_realm_instance.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/serializable_dynamic_node_invocation.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

using SerializableInstances =
    std::map<DynamicTensorSlot,
             std::pair<SerializableRealmInstance, SerializableRealmEvent>>;

/**
 * @brief Pull the instances for \p slots out of \p backing, keyed by slot.
 *
 * The values themselves are already on the wire inside the invocation, so the
 * instances travel keyed by the slot they are bound to rather than by the whole
 * value.
 */
static SerializableInstances instances_for_slots(
    std::map<DynamicTensorSlot, DynamicValueAttrs> const &slots,
    TensorInstanceBacking const &backing) {
  SerializableInstances result;
  for (auto const &[slot, value] : slots) {
    auto const &[instance, ready] = backing.backing.at(value);
    result.insert({slot,
                   std::pair{realm_instance_to_serializable(instance),
                             realm_event_to_serializable(ready)}});
  }
  return result;
}

/**
 * @brief Rebuild the value-keyed backing from the slot-keyed instances and the
 * values the invocation carries.
 *
 * \relates instances_for_slots
 */
static void restore_backing_for_slots(
    TensorInstanceBacking &backing,
    std::map<DynamicTensorSlot, DynamicValueAttrs> const &slots,
    SerializableInstances const &instances) {
  for (auto const &[slot, value] : slots) {
    auto const &[instance, ready] = instances.at(slot);
    backing.backing.insert(
        {value,
         std::pair{realm_instance_from_serializable(instance),
                   realm_event_from_serializable(ready)}});
  }
}

SerializableOpTaskArgs op_task_args_to_serializable(OpTaskArgs const &args) {
  return SerializableOpTaskArgs{
      /*invocation=*/dynamic_node_invocation_to_serializable(args.invocation),
      /*input_instances=*/
      instances_for_slots(args.invocation.inputs, args.tensor_backing),
      /*output_instances=*/
      instances_for_slots(args.invocation.outputs, args.tensor_backing),
      /*device_state=*/
      transform(args.device_state,
                device_specific_ptr_to_serializable<PerDeviceOpState>),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/device_specific_ptr_to_serializable(args.device_handle),
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

OpTaskArgs op_task_args_from_serializable(SerializableOpTaskArgs const &args) {
  DynamicNodeInvocation invocation =
      dynamic_node_invocation_from_serializable(args.invocation);

  TensorInstanceBacking tensor_backing = TensorInstanceBacking{{}};
  restore_backing_for_slots(
      tensor_backing, invocation.inputs, args.input_instances);
  restore_backing_for_slots(
      tensor_backing, invocation.outputs, args.output_instances);

  return OpTaskArgs{
      /*invocation=*/invocation,
      /*tensor_backing=*/tensor_backing,
      /*device_state=*/
      transform(args.device_state,
                device_specific_ptr_from_serializable<PerDeviceOpState>),
      /*profiling_settings=*/args.profiling_settings,
      /*device_handle=*/
      device_specific_ptr_from_serializable<ManagedPerDeviceFFHandle>(
          args.device_handle),
      /*optimizer_attrs=*/args.optimizer_attrs,
  };
}

} // namespace FlexFlow
