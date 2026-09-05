#include "kernels/gradient_reduction_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

template <DataType DT>
struct CPUGradientReductionKernel {
  void operator()(std::vector<GenericTensorAccessorR> const &inputs,
                  GenericTensorAccessorW const &output) {
    using T = real_type_t<DT>;
    size_t volume = get_num_elements(output.shape.dims).int_from_positive_int();

    T *out = output.get<DT>();
    for (size_t i = 0; i < volume; i++) {
      out[i] = inputs.at(0).get<DT>()[i];
    }
    for (size_t n = 1; n < inputs.size(); n++) {
      T const *in = inputs.at(n).get<DT>();
      for (size_t i = 0; i < volume; i++) {
        out[i] += in[i];
      }
    }
  }
};

void gradient_reduction_cpu_kernel(
    std::vector<GenericTensorAccessorR> const &inputs,
    GenericTensorAccessorW const &output) {
  DataTypeDispatch1<CPUGradientReductionKernel>{}(
      output.shape.data_type, inputs, output);
}

} // namespace FlexFlow
