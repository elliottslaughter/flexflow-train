#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FILL_TENSOR_ACCESSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_FILL_TENSOR_ACCESSOR_H

#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/device_stream_t.dtg.h"
#include "op-attrs/datatype_value.dtg.h"

namespace FlexFlow {

void fill_with_zeros(GenericTensorAccessorW const &accessor);

/**
 * \brief Zero every element of \p accessor, on \p stream.
 *
 * Unlike \ref fill_with_zeros this does not synchronize. The zeroing is
 * ordered against the rest of the work on \p stream and against nothing else,
 * which is what makes it usable from inside a task.
 */
void fill_with_zeros_on_stream(GenericTensorAccessorW const &accessor,
                               device_stream_t const &stream);

GenericTensorAccessorW create_accessor_w_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator);

GenericTensorAccessorR create_accessor_r_filled_with(
    TensorShape const &shape, DataTypeValue val, Allocator const &allocator);

} // namespace FlexFlow

#endif
