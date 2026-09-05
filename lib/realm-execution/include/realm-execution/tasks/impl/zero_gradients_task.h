#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_ZERO_GRADIENTS_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_ZERO_GRADIENTS_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "realm-execution/tensor_instance_backing.dtg.h"

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for recording which gradient
 * instances a node zeroes; see \ref spawn_zero_gradients_register_task.
 */
void zero_gradients_register_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief The function registered as a Realm task for zeroing a device's
 * gradients. Dispatched by \ref spawn_zero_gradients_task.
 */
void zero_gradients_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Tell the node that owns \p gradients that they are its to zero, once.
 */
Realm::Event
    spawn_zero_gradients_register_task(RealmContext &ctx,
                                       Realm::Processor target_proc,
                                       TensorInstanceBacking const &gradients,
                                       Realm::Event precondition);

/**
 * \brief Zero every gradient instance registered for \p target_proc's device.
 *
 * One task per device rather than a fill per instance: a fill is a Realm
 * operation of its own, and there are as many of them as there are gradients,
 * which is more operations than the whole backward pass has tasks. Inside a
 * task the same work is a \c cudaMemsetAsync per instance on the task's stream.
 */
Realm::Event spawn_zero_gradients_task(RealmContext &ctx,
                                       Realm::Processor target_proc,
                                       Realm::Event precondition);

} // namespace FlexFlow

#endif
