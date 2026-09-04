#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_WEIGHT_INIT_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_WEIGHT_INIT_TASK_H

#include "op-attrs/initializer_attrs.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for filling in a weight's
 * initial contents. Dispatched by \ref spawn_weight_init_task.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
void weight_init_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Launches the task (\ref weight_init_task_body) that fills one shard
 * of a weight with values drawn according to \p initializer.
 *
 * The task runs on the device that holds the shard, so a weight is written by
 * the node that owns it rather than by the controller. The values themselves
 * are generated on the target node from \p initializer and \p salt, so no
 * weight data crosses the network.
 *
 * \param value the (shard-expanded) weight tensor to fill
 * \param tensor_backing the instances for \p value, which must contain at
 *        least \p value itself
 * \param initializer the distribution to draw from
 * \param salt distinguishes weights sharing an \ref InitializerAttrs; it is
 *        computed on the controller so that every shard of a weight agrees.
 *        See \ref initialize_weight_shard.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
Realm::Event spawn_weight_init_task(RealmContext &ctx,
                                    Realm::Processor target_proc,
                                    DynamicValueAttrs const &value,
                                    TensorInstanceBacking const &tensor_backing,
                                    InitializerAttrs const &initializer,
                                    size_t salt,
                                    Realm::Event precondition);

} // namespace FlexFlow

#endif
