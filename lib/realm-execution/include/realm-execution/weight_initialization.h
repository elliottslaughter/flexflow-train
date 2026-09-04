#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_WEIGHT_INITIALIZATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_WEIGHT_INITIALIZATION_H

#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Launches tasks (using \ref spawn_weight_init_task) to fill the
 * instances backing the weights of \p g with values drawn from each weight's
 * \ref InitializerAttrs.
 *
 * Nothing else writes a weight before the first forward pass reads it, so
 * without this a model would train from whatever happened to be in the memory
 * its weights were allocated out of.
 *
 * One task is launched per shard, on the device that holds it, so no weight
 * data crosses the network: each task generates its weight from the
 * initializer locally. Unlike \ref
 * perform_distributed_per_device_op_state_initialization there is nothing to
 * share between the forward and backward invocations of a node, because only
 * the forward weight tensor holds the weight itself (see \ref
 * get_weight_initializers).
 *
 * \warning The tasks' events are left outstanding on \p ctx rather than waited
 * on here, so whatever consumes the weights afterwards has to take
 * \ref RealmContext::get_outstanding_events as a precondition.
 */
void perform_weight_initialization(
    RealmContext &ctx,
    DynamicOpenDataflowGraph const &g,
    TensorInstanceBacking const &tensor_instance_backing,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
