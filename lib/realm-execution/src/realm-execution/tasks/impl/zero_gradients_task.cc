#include "realm-execution/tasks/impl/zero_gradients_task.h"
#include "kernels/fill_tensor_accessor.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/tasks/impl/serializable_zero_gradients_register_args.dtg.h"
#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"
#include "task-spec/permissions.h"
#include "utils/optional.h"
#include <map>
#include <mutex>
#include <vector>

namespace FlexFlow {

namespace {

// Written by the registration task during initialization and read by the
// zeroing task afterwards, from whichever threads the runtime happens to use.
std::mutex gradients_mutex;
std::map<global_device_id_t, std::vector<GenericTensorAccessorW>>
    gradients_by_device;

} // namespace

void zero_gradients_register_task_body(void const *args,
                                       size_t arglen,
                                       void const *userdata,
                                       size_t userlen,
                                       Realm::Processor proc) {
  SerializableZeroGradientsRegisterArgs task_args =
      deserialize_task_args<SerializableZeroGradientsRegisterArgs>(args,
                                                                   arglen);
  TensorInstanceBacking gradients =
      tensor_instance_backing_from_serializable(task_args.gradients);

  RealmContext ctx{proc};
  std::vector<GenericTensorAccessorW> accessors;
  for (auto const &[value, instance] : gradients.backing) {
    accessors.push_back(dynamic_tensor_accessor_from_instance(
                            /*inst=*/instance.first,
                            /*ready=*/instance.second,
                            /*parallel_tensor_shape=*/
                            assert_unwrap(value.parallel_tensor_shape),
                            /*permissions=*/Permissions::WO,
                            /*for_processor=*/ctx.get_current_processor())
                            .require_write());
  }

  std::lock_guard<std::mutex> lock{gradients_mutex};
  gradients_by_device.insert_or_assign(ctx.get_current_global_device_id(),
                                       accessors);
}

void zero_gradients_task_body(void const *args,
                              size_t arglen,
                              void const *userdata,
                              size_t userlen,
                              Realm::Processor proc) {
  RealmContext ctx{proc};
  device_stream_t stream = ctx.get_current_device_stream();

  std::vector<GenericTensorAccessorW> const *accessors = nullptr;
  {
    std::lock_guard<std::mutex> lock{gradients_mutex};
    // std::map does not move its elements, so this stays valid after the lock
    // is dropped; nothing writes the table once the model is running.
    accessors = &gradients_by_device.at(ctx.get_current_global_device_id());
  }

  for (GenericTensorAccessorW const &accessor : *accessors) {
    fill_with_zeros_on_stream(accessor, stream);
  }
}

Realm::Event
    spawn_zero_gradients_register_task(RealmContext &ctx,
                                       Realm::Processor target_proc,
                                       TensorInstanceBacking const &gradients,
                                       Realm::Event precondition) {
  auto serialized_args =
      serialize_task_args(SerializableZeroGradientsRegisterArgs{
          /*gradients=*/tensor_instance_backing_to_serializable(gradients),
      });

  return ctx.spawn_task(target_proc,
                        task_id_t::ZERO_GRADIENTS_REGISTER_TASK_ID,
                        serialized_args.data(),
                        serialized_args.size(),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

Realm::Event spawn_zero_gradients_task(RealmContext &ctx,
                                       Realm::Processor target_proc,
                                       Realm::Event precondition) {
  return ctx.spawn_task(target_proc,
                        task_id_t::ZERO_GRADIENTS_TASK_ID,
                        nullptr,
                        0,
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
