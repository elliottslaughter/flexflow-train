#include "task-spec/dynamic_graph/weight_initialization.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/initializer_kernels.h"
#include "op-attrs/ops/weight_attrs.dtg.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_dims_coord.h"
#include "op-attrs/tensor_shape.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "utils/containers/values.h"
#include "utils/optional.h"
#include <vector>

namespace FlexFlow {

std::map<DynamicValueAttrs, InitializerAttrs>
    get_weight_initializers(DynamicOpenDataflowGraph const &g) {
  std::map<DynamicValueAttrs, InitializerAttrs> result;

  for (DynamicNodeInvocation const &invocation : g.invocations) {
    std::optional<TrainingOperationAttrs> const &op_attrs =
        invocation.node_attrs.op_attrs;
    if (!op_attrs.has_value() || !op_attrs.value().has<PCGOperatorAttrs>()) {
      continue;
    }

    PCGOperatorAttrs const &pcg_op_attrs =
        op_attrs.value().get<PCGOperatorAttrs>();
    if (!pcg_op_attrs.has<WeightAttrs>()) {
      continue;
    }
    InitializerAttrs initializer = pcg_op_attrs.get<WeightAttrs>().initializer;

    // Pass expansion gives a weight node a forward, a backward and an update
    // invocation, and the weight tensor appears among the values of more than
    // one of them. Only the forward tensor holds the weight itself: the others
    // are its gradient and the optimizer's state.
    for (DynamicValueAttrs const &value : values(invocation.outputs)) {
      if (value.role != mk_dynamic_tensor_role_fwd()) {
        continue;
      }
      result.insert(std::pair{value, initializer});
    }
  }

  return result;
}

/**
 * \brief Copy the block of \p full that belongs to the shard at \p shard_coord
 * into \p shard.
 */
template <DataType DT>
struct CopyShardOfTensor {
  void operator()(GenericTensorAccessorW const &shard,
                  GenericTensorAccessorR const &full,
                  ParallelTensorSpaceCoordinate const &shard_coord) const {
    TensorDims const &shard_dims = shard.shape.dims;

    for (TensorDimsCoord const &shard_idx :
         get_tensor_dims_coord_set(shard_dims)) {
      TensorDimsCoord full_idx = shard_idx;
      for (ff_dim_t dim : get_ff_dim_t_set(shard_dims)) {
        tensor_dims_coord_at_idx(full_idx, dim) =
            shard_coord.shard_components.at(dim) * dim_at_idx(shard_dims, dim) +
            tensor_dims_coord_at_idx(shard_idx, dim);
      }

      shard.at<DT>(shard_idx) = full.at<DT>(full_idx);
    }
  }
};

void initialize_weight_shard(GenericTensorAccessorW const &shard,
                             DynamicValueAttrs const &value,
                             InitializerAttrs const &initializer,
                             size_t salt) {
  ParallelTensorShape shape = assert_unwrap(value.parallel_tensor_shape);

  ASSERT(get_sum_degree(shape) == 1_p,
         "cannot initialize a weight that is partitioned along the sum "
         "dimension, as each shard would then have to hold a share of the "
         "weight rather than the weight itself",
         value.tensor_guid);

  TensorShape full_shape = get_reduced_shape(shape);
  TensorShape shard_shape = get_piece_shape(shape);

  if (shard_shape == full_shape) {
    // The common case: the weight is either not parallelized at all or only
    // replicated, so this shard holds the whole tensor and can be generated
    // straight into it.
    initialize_tensor(shard, initializer, salt);
    return;
  }

  std::vector<char> full_buffer = std::vector<char>(
      get_size_in_bytes(full_shape).unwrap_num_bytes().unwrap_nonnegative());
  GenericTensorAccessorW full =
      GenericTensorAccessorW{full_shape, full_buffer.data(), DeviceType::CPU};
  initialize_tensor_cpu(full, initializer, salt);

  std::vector<char> shard_buffer = std::vector<char>(
      get_size_in_bytes(shard_shape).unwrap_num_bytes().unwrap_nonnegative());
  GenericTensorAccessorW shard_staging =
      GenericTensorAccessorW{shard_shape, shard_buffer.data(), DeviceType::CPU};

  DataTypeDispatch1<CopyShardOfTensor>{}(
      shard_shape.data_type,
      shard_staging,
      read_only_accessor_from_write_accessor(full),
      assert_unwrap(value.shard_coord));

  copy_accessor_data_to_l_from_r(
      shard, read_only_accessor_from_write_accessor(shard_staging));
}

} // namespace FlexFlow
