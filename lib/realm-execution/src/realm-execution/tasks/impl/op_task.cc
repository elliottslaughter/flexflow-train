#include "realm-execution/tasks/impl/op_task.h"
#include "local-execution/task_execution.h"
#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/dynamic_tensor_accessor_from_instance.h"
#include "realm-execution/op_task_arg_registry.h"
#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/impl/serializable_op_task_launch_args.dtg.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/per_device_op_state.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/optional.h"
#include <type_traits>

namespace FlexFlow {

void op_task_body(void const *args,
                  size_t arglen,
                  void const *userdata,
                  size_t userlen,
                  Realm::Processor proc) {
  SerializableOpTaskLaunchArgs launch =
      deserialize_task_args<SerializableOpTaskLaunchArgs>(args, arglen);

  // Everything that is a property of the graph was sent to this node once,
  // during initialization; a launch only names it. See \ref
  // register_op_task_args.
  OpTaskArgs task_args = get_registered_op_task_args(launch.invocation_id);
  task_args.profiling_settings = launch.profiling_settings;
  task_args.optimizer_attrs = launch.optimizer_attrs;

  RealmContext ctx{proc};
  device_handle_t device_handle =
      device_handle_t_from_device_specific_managed_ff_handle(
          task_args.device_handle, ctx.get_current_global_device_id());

  // The invocation already carries its accessors; they were resolved once,
  // when these arguments were registered.
  DynamicNodeInvocation const &invocation = task_args.invocation;

  execute_dynamic_node_invocation(
      /*invocation=*/invocation,
      /*allocator=*/ctx.get_current_device_allocator(),
      /*profiling_settings=*/task_args.profiling_settings,
      /*ff_handle=*/device_handle,
      /*per_device_op_state=*/
      transform(and_then(task_args.device_state,
                         [&](DeviceSpecificPtr<PerDeviceOpState> const &d) {
                           return d.get(ctx.get_current_global_device_id());
                         }),
                [](PerDeviceOpState *ptr) { return *ptr; }),
      /*optimizer_attrs=*/task_args.optimizer_attrs,
      /*global_device_id=*/ctx.get_current_global_device_id(),
      /*stream=*/ctx.get_current_device_stream());
}

Realm::Event spawn_op_task(RealmContext &ctx,
                           Realm::Processor target_proc,
                           DynamicNodeInvocation const &invocation,
                           dynamic_invocation_id_t const &invocation_id,
                           ProfilingSettings const &profiling_settings,
                           std::optional<OptimizerAttrs> const &optimizer_attrs,
                           Realm::Event precondition) {
  auto serialized_args = serialize_task_args(SerializableOpTaskLaunchArgs{
      /*invocation_id=*/invocation_id,
      /*profiling_settings=*/profiling_settings,
      /*optimizer_attrs=*/optimizer_attrs,
  });

  return ctx.spawn_task(
      target_proc,
      assert_unwrap(get_task_id_for_op(invocation.node_attrs, optimizer_attrs)),
      serialized_args.data(),
      serialized_args.size(),
      Realm::ProfilingRequestSet{},
      precondition);
}

} // namespace FlexFlow
