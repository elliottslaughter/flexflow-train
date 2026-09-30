#include "op-attrs/activation.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

OperatorType
  op_type_for_activation(Activation activation)
{
  switch (activation) {
    case Activation::RELU:
      return OperatorType::RELU;
    case Activation::SIGMOID:
      return OperatorType::SIGMOID;
    case Activation::TANH:
      return OperatorType::TANH;
    case Activation::GELU:
      return OperatorType::GELU;
    case Activation::SILU:
      return OperatorType::SILU;
    default:
      PANIC("Unknown activation", activation);
  }
}

} // namespace FlexFlow
