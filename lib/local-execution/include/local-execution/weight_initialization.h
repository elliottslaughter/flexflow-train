#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_WEIGHT_INITIALIZATION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_WEIGHT_INITIALIZATION_H

#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"

namespace FlexFlow {

/**
 * @brief Fill the tensors backing the weights of \p g with values drawn from
 * each weight's \ref InitializerAttrs.
 *
 * Nothing else writes a weight before the first forward pass reads it, so
 * without this a model would train from whatever happened to be in the memory
 * its weights were allocated out of.
 *
 * \note \p g must already have had its tensors allocated (see \ref
 * perform_tensor_allocation).
 */
void perform_weight_initialization(DynamicOpenDataflowGraph const &g);

} // namespace FlexFlow

#endif
