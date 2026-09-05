#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_FUSED_OP_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_FUSED_OP_TASK_H

#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"
#include <optional>

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for a fused group of
 * operator tasks. Dispatched by \ref spawn_fused_op_task.
 *
 * Runs the group's members' task bodies back to back, in the order they were
 * registered in. They all submit their work to the one stream this task was
 * given, and work submitted to a single stream runs in order, so the members'
 * dependencies on one another hold without any further synchronization.
 *
 * \see group_invocations_for_fusion for which invocations end up here.
 */
void fused_op_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Launch \ref fused_op_task_body for the group registered under
 * \p group_id.
 */
Realm::Event
    spawn_fused_op_task(RealmContext &ctx,
                        Realm::Processor target_proc,
                        dynamic_invocation_id_t const &group_id,
                        std::optional<OptimizerAttrs> const &optimizer_attrs,
                        Realm::Event precondition);

} // namespace FlexFlow

#endif
