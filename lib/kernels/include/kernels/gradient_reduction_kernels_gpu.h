#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GRADIENT_REDUCTION_KERNELS_GPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GRADIENT_REDUCTION_KERNELS_GPU_H

#include "kernels/accessor.h"
#include "kernels/device.h"
#include <vector>

namespace FlexFlow {

void gradient_reduction_gpu_kernel(
    ffStream_t stream,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output);

} // namespace FlexFlow

#endif
