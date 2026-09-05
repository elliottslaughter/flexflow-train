#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GRADIENT_REDUCTION_KERNELS_CPU_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_GRADIENT_REDUCTION_KERNELS_CPU_H

#include "kernels/accessor.h"
#include <vector>

namespace FlexFlow {

void gradient_reduction_cpu_kernel(
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output);

} // namespace FlexFlow

#endif
