#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZERS_KAIMING_INITIALIZER_NONLINEARITY_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_INITIALIZERS_KAIMING_INITIALIZER_NONLINEARITY_H

#include "op-attrs/initializers/kaiming_initializer_nonlinearity.dtg.h"
#include <optional>

namespace FlexFlow {

/**
 * @brief The recommended gain value for a given nonlinearity, from pytorch
 *
 * see
 * https://github.com/pytorch/pytorch/blob/bd019c0bb485904a99fb38589444b1461ab1e486/torch/nn/init.py#L72-L139
 */
float gain_for_nonlinearity(
    KaimingInitializerNonlinearity nonlinearity,
    std::optional<float> const &negative_slope = std::nullopt);

} // namespace FlexFlow

#endif
