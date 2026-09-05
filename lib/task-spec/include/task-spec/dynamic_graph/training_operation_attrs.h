#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_TRAINING_OPERATION_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_TRAINING_OPERATION_ATTRS_H

#include "op-attrs/operator_type.dtg.h"
#include "task-spec/dynamic_graph/training_op_type.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"

namespace FlexFlow {

bool training_op_attrs_has_op_type(TrainingOperationAttrs const &,
                                   OperatorType);
TrainingOpType training_op_attrs_get_op_type(TrainingOperationAttrs const &);

/**
 * \brief Whether this operator's backward pass writes every element of the
 * gradients it produces, rather than accumulating into them.
 *
 * A gradient whose only writer overwrites it does not have to be cleared before
 * the backward pass, which is most of them: an operator that reads a tensor
 * once produces its whole gradient, and where a tensor is read more than once
 * the subgradients get instances of their own and are summed by a gradient
 * reduction. What is left is the operators that scatter -- whose backward pass
 * reaches only the elements some index selected, and leaves the rest of the
 * gradient as it found it -- and the reductions themselves, which fold into
 * their destination and so need it to start at zero.
 *
 * \warning This has to agree with what the kernels actually do. A kernel that
 * accumulates while this says it overwrites reads a gradient that was never
 * cleared.
 */
bool bwd_task_overwrites_grads(TrainingOperationAttrs const &);

} // namespace FlexFlow

#endif
