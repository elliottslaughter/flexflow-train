#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_GROUP_REGISTER_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_OP_TASK_GROUP_REGISTER_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"
#include <vector>

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for putting a fused group
 * into the node-local table read by \ref fused_op_task_body.
 */
void op_task_group_register_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Tell the node that will be running \p group_id which invocations the
 * group runs, once, so that later launches of it need only name it.
 *
 * \warning Every member's own arguments must already have been registered on
 * that node; see \ref spawn_op_task_arg_register_task.
 */
Realm::Event spawn_op_task_group_register_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    dynamic_invocation_id_t const &group_id,
    std::vector<dynamic_invocation_id_t> const &member_ids,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
