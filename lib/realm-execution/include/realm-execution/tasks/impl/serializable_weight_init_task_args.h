#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_WEIGHT_INIT_TASK_ARGS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_WEIGHT_INIT_TASK_ARGS_H

#include "realm-execution/tasks/impl/serializable_weight_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/weight_init_task_args.dtg.h"

namespace FlexFlow {

SerializableWeightInitTaskArgs
    weight_init_task_args_to_serializable(WeightInitTaskArgs const &);
WeightInitTaskArgs weight_init_task_args_from_serializable(
    SerializableWeightInitTaskArgs const &);

} // namespace FlexFlow

#endif
