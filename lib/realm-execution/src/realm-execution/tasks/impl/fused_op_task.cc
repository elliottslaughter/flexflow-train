#include "realm-execution/tasks/impl/fused_op_task.h"
#include "local-execution/task_execution.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/op_task_arg_registry.h"
#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_op_task_launch_args.dtg.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"

namespace FlexFlow {

void fused_op_task_body(void const *args,
                        size_t arglen,
                        void const *userdata,
                        size_t userlen,
                        Realm::Processor proc) {
  SerializableOpTaskLaunchArgs launch =
      deserialize_task_args<SerializableOpTaskLaunchArgs>(args, arglen);

  std::vector<OpTaskArgs const *> const &members =
      get_registered_op_task_group(launch.invocation_id);

  // Everything below is the same for every member -- that they share a device
  // is one of the conditions for being fused together -- so it is set up once
  // rather than once per member. That is most of the point: what a fused task
  // saves over separate tasks is everything but the kernel launches.
  RealmContext ctx{proc};
  global_device_id_t global_device_id = ctx.get_current_global_device_id();
  Allocator &allocator = ctx.get_current_device_allocator();
  device_stream_t stream = ctx.get_current_device_stream();
  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_ff_handle(
          members.front()->device_handle, global_device_id);

  for (OpTaskArgs const *member : members) {
    execute_dynamic_node_invocation(
        /*invocation=*/member->invocation,
        /*allocator=*/allocator,
        /*profiling_settings=*/std::nullopt,
        /*ff_handle=*/device_handle,
        /*per_device_op_state=*/
        transform(and_then(member->device_state,
                           [&](DeviceSpecificPtr<PerDeviceOpState> const &d) {
                             return d.get(global_device_id);
                           }),
                  [](PerDeviceOpState *ptr) { return *ptr; }),
        /*optimizer_attrs=*/launch.optimizer_attrs,
        /*global_device_id=*/global_device_id,
        /*stream=*/stream);
  }
}

Realm::Event
    spawn_fused_op_task(RealmContext &ctx,
                        Realm::Processor target_proc,
                        dynamic_invocation_id_t const &group_id,
                        std::optional<OptimizerAttrs> const &optimizer_attrs,
                        Realm::Event precondition) {
  auto serialized_args = serialize_task_args(SerializableOpTaskLaunchArgs{
      /*invocation_id=*/group_id,
      /*optimizer_attrs=*/optimizer_attrs,
  });

  return ctx.spawn_task(target_proc,
                        task_id_t::FUSED_OP_TASK_ID,
                        serialized_args.data(),
                        serialized_args.size(),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
