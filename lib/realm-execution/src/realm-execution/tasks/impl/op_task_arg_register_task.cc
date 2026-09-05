#include "realm-execution/tasks/impl/op_task_arg_register_task.h"
#include "realm-execution/op_task_arg_registry.h"
#include "realm-execution/tasks/impl/serializable_op_task_arg_register_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_op_task_args.h"
#include "realm-execution/tasks/serializer/task_arg_serializer.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

void op_task_arg_register_task_body(void const *args,
                                    size_t arglen,
                                    void const *userdata,
                                    size_t userlen,
                                    Realm::Processor proc) {
  SerializableOpTaskArgRegisterArgs task_args =
      deserialize_task_args<SerializableOpTaskArgRegisterArgs>(args, arglen);

  register_op_task_args(task_args.invocation_id,
                        op_task_args_from_serializable(task_args.args));
}

Realm::Event spawn_op_task_arg_register_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    dynamic_invocation_id_t const &invocation_id,
    OpTaskArgs const &args,
    Realm::Event precondition) {
  auto serialized_args =
      serialize_task_args(SerializableOpTaskArgRegisterArgs{
          /*invocation_id=*/invocation_id,
          /*args=*/op_task_args_to_serializable(args),
      });

  return ctx.spawn_task(target_proc,
                        task_id_t::OP_TASK_ARG_REGISTER_TASK_ID,
                        serialized_args.data(),
                        serialized_args.size(),
                        Realm::ProfilingRequestSet{},
                        precondition);
}

} // namespace FlexFlow
