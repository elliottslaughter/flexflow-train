#include "kernels/gradient_reduction_kernels.h"
#include "kernels/gradient_reduction_kernels_cpu.h"
#include "kernels/gradient_reduction_kernels_gpu.h"
#include "utils/exception.h"

namespace FlexFlow {

void gradient_reduction_kernel(
    device_stream_t const &stream,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output) {
  ASSERT(!inputs.empty(), "a gradient reduction has no subgradients to sum");
  for (GenericTensorAccessorR const &input : inputs) {
    ASSERT(input.shape == output.shape,
           "a gradient reduction's subgradients must all have the shape of its "
           "output",
           input.shape,
           output.shape);
  }

  if (stream.is_gpu()) {
    gradient_reduction_gpu_kernel(
        /*stream=*/stream.require_gpu(),
        /*inputs=*/inputs,
        /*output=*/output);
  } else {
    ASSERT(stream.is_cpu());
    gradient_reduction_cpu_kernel(
        /*inputs=*/inputs,
        /*output=*/output);
  }
}

} // namespace FlexFlow
