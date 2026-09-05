#include "realm-execution/tasks/impl/op_task_group_register_task.h"
#include "realm-execution/op_task_arg_registry.h"
#include "realm-execution/tasks/impl/serializable_op_task_group_register_args.dtg.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

void op_task_group_register_task_body(void const *args,
                                      size_t arglen,
                                      void const *userdata,
                                      size_t userlen,
                                      Realm::Processor proc) {
  SerializableOpTaskGroupRegisterArgs task_args =
      deserialize_task_args<SerializableOpTaskGroupRegisterArgs>(args, arglen);

  register_op_task_group(task_args.group_id, task_args.member_ids);
}

Realm::Event spawn_op_task_group_register_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    dynamic_invocation_id_t const &group_id,
    std::vector<dynamic_invocation_id_t> const &member_ids,
    Realm::Event precondition) {
  auto serialized_args =
      serialize_task_args(SerializableOpTaskGroupRegisterArgs{
          /*group_id=*/group_id,
          /*member_ids=*/member_ids,
      });

  return ctx.spawn_task(target_proc,
                        task_id_t::OP_TASK_GROUP_REGISTER_TASK_ID,
                        serialized_args.data(),
                        serialized_args.size(),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
