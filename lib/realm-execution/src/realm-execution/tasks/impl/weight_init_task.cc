#include "realm-execution/tasks/impl/weight_init_task.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/tasks/impl/serializable_weight_init_task_args.h"
#include "realm-execution/tasks/impl/weight_init_task_args.dtg.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/weight_initialization.h"
#include "task-spec/permissions.h"
#include "utils/optional.h"

namespace FlexFlow {

void weight_init_task_body(void const *args,
                           size_t arglen,
                           void const *userdata,
                           size_t userlen,
                           Realm::Processor proc) {
  WeightInitTaskArgs task_args = weight_init_task_args_from_serializable(
      deserialize_task_args<SerializableWeightInitTaskArgs>(args, arglen));

  RealmContext ctx{proc};

  DynamicValueAttrs const &value = task_args.value;
  auto const &[inst, ready] = task_args.tensor_backing.backing.at(value);
  GenericTensorAccessorW shard = dynamic_tensor_accessor_from_instance(
                                     inst,
                                     ready,
                                     assert_unwrap(value.parallel_tensor_shape),
                                     Permissions::WO,
                                     ctx.get_current_processor())
                                     .require_write();

  initialize_weight_shard(shard,
                          value,
                          task_args.initializer,
                          ctx.get_current_device_stream(),
                          task_args.salt);
}

Realm::Event spawn_weight_init_task(RealmContext &ctx,
                                    Realm::Processor target_proc,
                                    DynamicValueAttrs const &value,
                                    TensorInstanceBacking const &tensor_backing,
                                    InitializerAttrs const &initializer,
                                    size_t salt,
                                    Realm::Event precondition) {
  WeightInitTaskArgs task_args = WeightInitTaskArgs{
      value,
      tensor_backing,
      initializer,
      salt,
  };

  std::string serialized_args =
      serialize_task_args(weight_init_task_args_to_serializable(task_args));

  return ctx.spawn_task(target_proc,
                        task_id_t::WEIGHT_INIT_TASK_ID,
                        serialized_args.data(),
                        serialized_args.size(),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
