#ifndef _FLEXFLOW_KERNELS_INCLUDE_KERNELS_INITIALIZER_KERNELS_H
#define _FLEXFLOW_KERNELS_INCLUDE_KERNELS_INITIALIZER_KERNELS_H

#include "accessor.h"
#include "kernels/cpu.h"
#include "op-attrs/datatype_value.dtg.h"
#include "op-attrs/initializer_attrs.dtg.h"

namespace FlexFlow {

/**
 * \brief Fill \p tensor with values drawn according to \p attrs.
 *
 * \p tensor must be CPU-resident; see \ref initialize_tensor for a version
 * that works no matter where the tensor lives.
 *
 * \param salt distinguishes tensors that share an \ref InitializerAttrs. The
 * random initializers seed their generator with the seed held in \p attrs
 * combined with \p salt, so two tensors of the same shape drawn with the same
 * attrs and the same salt come out bit-identical. That is rarely what is
 * wanted: the default initializers op-attrs picks all carry a seed of 0, so
 * leaving the salt at 0 gives every layer of a given shape the same weights.
 * Pass something that identifies the tensor.
 *
 * \note The generator is a \c std::mt19937 seeded through a \c std::seed_seq,
 * and the distributions are computed by hand rather than with \c
 * std::uniform_real_distribution and friends, which are not specified to
 * produce the same sequence across standard library implementations. A given
 * (attrs, salt, shape) therefore always produces the same contents.
 */
void initialize_tensor_cpu(GenericTensorAccessorW const &tensor,
                           InitializerAttrs const &attrs,
                           size_t salt = 0);

/**
 * \brief Fill \p tensor with values drawn according to \p attrs, whether
 * \p tensor lives on the host or on a device.
 *
 * The initializers themselves run on the host, so a device-resident tensor is
 * filled by generating its contents into a staging buffer and copying them
 * over.
 *
 * \relates initialize_tensor_cpu
 */
void initialize_tensor(GenericTensorAccessorW const &tensor,
                       InitializerAttrs const &attrs,
                       size_t salt = 0);

void zero_init_kernel(TaskLocation const &, GenericTensorAccessorW const &);
void zero_init_kernel_gpu(GenericTensorAccessorW const &);
void zero_init_kernel_cpu(GenericTensorAccessorW const &);

void constant_init_kernel(TaskLocation const &,
                          GenericTensorAccessorW const &,
                          DataTypeValue);
void constant_init_kernel_gpu(GenericTensorAccessorW const &, DataTypeValue);
void constant_init_kernel_cpu(GenericTensorAccessorW const &, DataTypeValue);

} // namespace FlexFlow

#endif
