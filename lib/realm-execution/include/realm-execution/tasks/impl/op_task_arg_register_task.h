#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_ARG_REGISTER_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_ARG_REGISTER_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for putting a task's
 * arguments into the node-local \ref register_op_task_args table.
 */
void op_task_arg_register_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Send \p args to the node that will be running \p invocation_id, once,
 * so that later launches of it need only name it.
 */
Realm::Event
    spawn_op_task_arg_register_task(RealmContext &ctx,
                                    Realm::Processor target_proc,
                                    dynamic_invocation_id_t const &invocation_id,
                                    OpTaskArgs const &args,
                                    Realm::Event precondition);

} // namespace FlexFlow

#endif
