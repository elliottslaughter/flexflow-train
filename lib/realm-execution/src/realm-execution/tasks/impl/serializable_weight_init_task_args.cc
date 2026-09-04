#include "realm-execution/tasks/impl/serializable_weight_init_task_args.h"
#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"

namespace FlexFlow {

SerializableWeightInitTaskArgs
    weight_init_task_args_to_serializable(WeightInitTaskArgs const &args) {
  return SerializableWeightInitTaskArgs{
      /*value=*/dynamic_value_attrs_to_serializable(args.value),
      /*tensor_backing=*/
      tensor_instance_backing_to_serializable(args.tensor_backing),
      /*initializer=*/args.initializer,
      /*salt=*/args.salt,
  };
}

WeightInitTaskArgs weight_init_task_args_from_serializable(
    SerializableWeightInitTaskArgs const &args) {
  return WeightInitTaskArgs{
      /*value=*/dynamic_value_attrs_from_serializable(args.value),
      /*tensor_backing=*/
      tensor_instance_backing_from_serializable(args.tensor_backing),
      /*initializer=*/args.initializer,
      /*salt=*/args.salt,
  };
}

} // namespace FlexFlow
