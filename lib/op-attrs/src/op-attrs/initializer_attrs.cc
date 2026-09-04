#include "op-attrs/initializer_attrs.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/initializers/kaiming_initializer_nonlinearity.h"
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

InitializerAttrs make_zero_initializer() {
  return InitializerAttrs{ZeroInitializerAttrs{}};
}

// from pytorch:
// see
// https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L456-L518
InitializerAttrs
    make_kaiming_uniform(TensorDims const &dims,
                         float a,
                         KaimingInitializerMode mode,
                         KaimingInitializerNonlinearity nonlinearity,
                         int seed) {

  positive_int fan = calculate_fan_for_mode(dims, mode);
  float gain = gain_for_nonlinearity(nonlinearity, a);
  float std = gain / sqrtf(static_cast<float>(fan.int_from_positive_int()));
  float bound = sqrtf(3.0) * std;

  return InitializerAttrs{UniformInitializerAttrs{
      /*seed=*/seed,
      /*min_val=*/-bound,
      /*max_val=*/bound,
  }};
}

} // namespace FlexFlow
