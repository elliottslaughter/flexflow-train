#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/tensor_dims.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

positive_int calculate_fan_for_mode(TensorDims const &dims,
                                    KaimingInitializerMode mode) {
  ASSERT(get_num_dims(dims) >= 2_n,
         "fan_in and fan_out cannot be computed for a tensor with fewer than "
         "two dimensions",
         dims);

  // A weight is indexed as [out, in, ...], so it is dimension 1 that gives
  // fan_in and dimension 0 that gives fan_out.
  positive_int num_output_fmaps = dim_at_idx(dims, relative_ff_dim_t{0});
  positive_int num_input_fmaps = dim_at_idx(dims, relative_ff_dim_t{1});

  positive_int receptive_field_size = get_num_elements(
      slice_tensor_dims(dims, relative_ff_dim_t{2}, std::nullopt));

  if (mode == KaimingInitializerMode::FAN_IN) {
    return num_input_fmaps * receptive_field_size;
  } else {
    ASSERT(mode == KaimingInitializerMode::FAN_OUT);

    return num_output_fmaps * receptive_field_size;
  }
}

} // namespace FlexFlow
