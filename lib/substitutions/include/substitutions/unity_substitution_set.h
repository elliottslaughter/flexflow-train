#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNITY_SUBSTITUTION_SET_H

#include "op-attrs/activation.dtg.h"
#include "op-attrs/ops/batch_norm_mode.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "substitutions/substitution.dtg.h"
#include "utils/fmt/vector.h"
#include <random>

namespace FlexFlow {

std::optional<Substitution>
    get_random_substitution(std::mt19937 &,
                            MachineComputeSpecification const &resources);

std::vector<Substitution>
    get_unity_substitution_set(MachineComputeSpecification const &resources);

std::vector<Substitution>
    get_expanded_substitution_set(MachineComputeSpecification const &resources);

Substitution create_replicate_linear_combine(positive_int num_dims,
                                             positive_int degree,
                                             bool use_bias);
Substitution create_partition_linear_combine(positive_int num_dims,
                                             positive_int degree,
                                             bool use_bias);
Substitution create_partition_conv2d_combine(positive_int num_dims,
                                             positive_int degree);
Substitution create_partition_attention_combine(positive_int num_heads,
                                                positive_int degree);
Substitution create_replicate_attention_reduce(positive_int num_heads,
                                               positive_int degree);
Substitution create_partition_add_combine(ff_dim_t parallel_dim,
                                          positive_int degree);
Substitution create_partition_relu_combine(ff_dim_t parallel_dim,
                                           positive_int degree);
Substitution create_partition_softmax_combine(ff_dim_t softmax_dim,
                                              ff_dim_t partition_dim,
                                              positive_int degree);
Substitution create_fuse_linear_activation(Activation activation);

/**
 * \brief Fold an elementwise \p activation into the affine batch norm in mode
 * \p mode whose output it reads, so that one kernel computes both.
 *
 * \details The compile-time counterpart of \ref perform_operation_fusion.
 * Only offered by \ref get_expanded_substitution_set for the pairs \ref
 * batch_norm_supports_fused_activation and \ref
 * batch_norm_mode_supports_fused_activation allow.
 *
 * \note The pattern does not require the batch norm's output to have no other
 * readers, which \ref perform_operation_fusion does check.
 */
Substitution create_fuse_batch_norm_activation(Activation activation,
                                               BatchNormMode mode);

} // namespace FlexFlow

#endif
