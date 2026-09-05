#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_ESTIMATOR_LOCAL_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_COST_ESTIMATOR_LOCAL_COST_ESTIMATOR_H

#include "compiler/cost_estimator/cost_estimator.h"
#include "kernels/allocation.h"
#include "kernels/device_handle_t.dtg.h"
#include "kernels/device_stream_t.dtg.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/machine_interconnect_specification.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"
#include <optional>

namespace FlexFlow {

struct LocalCostEstimator : public ICostEstimator {
  explicit LocalCostEstimator(
      MachineInterconnectSpecification const &interconnect_specification,
      Allocator &allocator,
      ProfilingSettings const &profiling_settings,
      device_handle_t const &device_handle,
      global_device_id_t device_idx);

  LocalCostEstimator(LocalCostEstimator const &) = delete;
  LocalCostEstimator(LocalCostEstimator &&) = delete;
  ~LocalCostEstimator() = default;

  OpCostMetrics estimate_cost(OpCostEstimateKey const &) const override;

  milliseconds_t estimate_cost(TensorSetMovement const &) const override;

private:
  MachineInterconnectSpecification interconnect_specification;
  Allocator allocator;
  ProfilingSettings profiling_settings;
  device_handle_t device_handle;
  global_device_id_t device_idx;

  // Nothing is running these tasks for us here, so this is where the stream
  // they run on comes from. Held for the estimator's lifetime rather than made
  // per call: a CUDA stream is a resource, and creating one per kernel launch
  // is what used to exhaust the device.
  std::optional<ManagedFFStream> owned_stream;
  device_stream_t stream;
};
CHECK_RC_COPY_VIRTUAL_COMPLIANT(LocalCostEstimator);

CostEstimator get_local_cost_estimator(
    MachineInterconnectSpecification const &interconnect_specification,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    global_device_id_t device_idx);

} // namespace FlexFlow

#endif
