#include "internal/device.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/gradient_reduction_kernels_gpu.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

template <typename T>
__global__ void assign_from_kernel(size_t volume, T const *input, T *output) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] = input[i];
  }
}

template <typename T>
__global__ void
    accumulate_into_kernel(size_t volume, T const *input, T *output) {
  CUDA_KERNEL_LOOP(i, volume) {
    output[i] += input[i];
  }
}

template <DataType DT>
struct GPUGradientReductionKernel {
  void operator()(ffStream_t stream,
                  std::vector<GenericTensorAccessorR> const &inputs,
                  GenericTensorAccessorW const &output) {
    using T = real_type_t<DT>;
    size_t volume = get_num_elements(output.shape.dims).int_from_positive_int();
    int blocks = GET_BLOCKS(volume);

    // The first subgradient is copied rather than added, so that the output
    // does not have to have been cleared.
    assign_from_kernel<T><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
        volume, inputs.at(0).get<DT>(), output.get<DT>());
    for (size_t i = 1; i < inputs.size(); i++) {
      accumulate_into_kernel<T><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
          volume, inputs.at(i).get<DT>(), output.get<DT>());
    }
  }
};

void gradient_reduction_gpu_kernel(
    ffStream_t stream,
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<GPUGradientReductionKernel>{}(
      output.shape.data_type, stream, inputs, output);
}

} // namespace FlexFlow
