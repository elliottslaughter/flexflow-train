#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_WEIGHT_INITIALIZATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_WEIGHT_INITIALIZATION_H

#include "kernels/accessor.h"
#include "op-attrs/initializer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include <map>

namespace FlexFlow {

/**
 * \brief Map each forward weight value of \p g to the \ref InitializerAttrs
 * that its contents should be drawn from.
 *
 * A weight is the only kind of tensor whose starting contents are part of the
 * model rather than something the runtime is handed, and a weight node has no
 * task of its own in the forward pass, so nothing in the execution pipeline
 * ever looks at \ref WeightAttrs::initializer. It is up to the backend to
 * fill weights in once it has allocated storage for them, and this is how it
 * finds out what to fill them with.
 *
 * \note \p g must already have been pass expanded (see \ref
 * perform_pass_expansion), as that is what tells the forward weight tensor
 * apart from its gradient and its optimizer state.
 */
std::map<DynamicValueAttrs, InitializerAttrs>
    get_weight_initializers(DynamicOpenDataflowGraph const &g);

/**
 * \brief Fill \p shard, the piece of the weight \p value that lives on this
 * device, with values drawn according to \p initializer.
 *
 * A weight that is sharded across devices is generated whole and then sliced,
 * so that its contents do not depend on how it was parallelized: the
 * initializer describes the tensor as the model declares it, not whatever piece
 * happens to land on one device. Replicated shards therefore all come out
 * identical, as they must.
 *
 * \param salt distinguishes weights that share an \ref InitializerAttrs, and
 * has to be the same on every device holding a piece of this weight. See \ref
 * initialize_tensor_cpu.
 */
void initialize_weight_shard(GenericTensorAccessorW const &shard,
                             DynamicValueAttrs const &value,
                             InitializerAttrs const &initializer,
                             size_t salt);

} // namespace FlexFlow

#endif
