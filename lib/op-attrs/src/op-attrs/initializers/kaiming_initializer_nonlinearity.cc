#include "op-attrs/initializers/kaiming_initializer_nonlinearity.h"
#include <cmath>
#include <libassert/assert.hpp>

namespace FlexFlow {

float gain_for_nonlinearity(KaimingInitializerNonlinearity nonlinearity,
                            std::optional<float> const &negative_slope) {
  if (nonlinearity == KaimingInitializerNonlinearity::RELU) {
    ASSERT(!negative_slope.has_value());
    return sqrtf(2.0);
  } else {
    ASSERT(nonlinearity == KaimingInitializerNonlinearity::LEAKY_RELU);

    return sqrtf(2.0 / (1 + negative_slope.value() * negative_slope.value()));
  }
}

} // namespace FlexFlow
