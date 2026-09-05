#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GRADIENT_REDUCTION_KERNELS_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GRADIENT_REDUCTION_KERNELS_H

#include "kernels/accessor.h"
#include "kernels/device_stream_t.dtg.h"
#include <vector>

namespace FlexFlow {

/**
 * \brief Sum \p inputs into \p output.
 *
 * The backward pass of a value that was read more than once produces one
 * subgradient per reader, each with an instance of its own, and this is what
 * adds them back together.
 *
 * \note \p output is written rather than accumulated into, so it does not have
 * to be cleared first. See \ref bwd_task_overwrites_grads.
 */
void gradient_reduction_kernel(
    device_stream_t const &stream,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output);

} // namespace FlexFlow

#endif
