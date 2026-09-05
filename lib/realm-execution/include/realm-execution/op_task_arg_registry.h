#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_OP_TASK_ARG_REGISTRY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_OP_TASK_ARG_REGISTRY_H

#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"

namespace FlexFlow {

/**
 * @brief A node-local table of the task arguments that were shipped to this
 * node once, during initialization.
 *
 * Almost everything a task is told is a property of the graph, not of the
 * iteration, and the graph does not change once a \ref PCGInstance exists. It
 * therefore only has to cross to the node that will run the task once, rather
 * than being encoded, sent and decoded again on every one of the thousands of
 * launches per iteration. What a launch then carries is the invocation's id and
 * the handful of things that really do vary.
 *
 * \warning Ids are only meaningful for the graph they were computed from, so
 * this table is only valid for the \ref PCGInstance that filled it in.
 */
void register_op_task_args(dynamic_invocation_id_t const &invocation_id,
                           OpTaskArgs const &args);

OpTaskArgs const &
    get_registered_op_task_args(dynamic_invocation_id_t const &invocation_id);

} // namespace FlexFlow

#endif
