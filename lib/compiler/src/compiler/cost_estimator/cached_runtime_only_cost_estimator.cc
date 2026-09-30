#include "compiler/cost_estimator/cached_runtime_only_cost_estimator.h"
#include "utils/containers/contains_key.h"

namespace FlexFlow {

CachedRuntimeOnlyCostEstimator::CachedRuntimeOnlyCostEstimator(
    RuntimeOnlyCostEstimator const &underlying)
    : underlying(underlying) {}

RuntimeOnlyOpCostMetrics CachedRuntimeOnlyCostEstimator::estimate_cost(
    RuntimeOnlyOpCostEstimateKey const &key) const {
  if (!contains_key(this->op_costs, key)) {
    this->op_costs.insert({key, this->underlying.estimate_cost(key)});
  }
  return this->op_costs.at(key);
}

milliseconds_t CachedRuntimeOnlyCostEstimator::estimate_cost(
    TensorSetMovement const &tensor_set_movement) const {
  return this->underlying.estimate_cost(tensor_set_movement);
}

RuntimeOnlyCostEstimator make_cached_runtime_only_cost_estimator(
    RuntimeOnlyCostEstimator const &underlying) {
  return RuntimeOnlyCostEstimator::create<CachedRuntimeOnlyCostEstimator>(
      underlying);
}

} // namespace FlexFlow
