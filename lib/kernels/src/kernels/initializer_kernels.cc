#include "kernels/initializer_kernels.h"
#include "kernels/accessor.h"
#include "op-attrs/tensor_shape.h"
#include <vector>

namespace FlexFlow {

void initialize_tensor(GenericTensorAccessorW const &tensor,
                       InitializerAttrs const &attrs,
                       size_t salt) {
  if (tensor.device_type == DeviceType::CPU) {
    initialize_tensor_cpu(tensor, attrs, salt);
    return;
  }

  std::vector<char> staging_buffer = std::vector<char>(
      get_size_in_bytes(tensor.shape).unwrap_num_bytes().unwrap_nonnegative());
  GenericTensorAccessorW staging = GenericTensorAccessorW{
      tensor.shape, staging_buffer.data(), DeviceType::CPU};

  initialize_tensor_cpu(staging, attrs, salt);
  copy_accessor_data_to_l_from_r(
      tensor, read_only_accessor_from_write_accessor(staging));
}

} // namespace FlexFlow
