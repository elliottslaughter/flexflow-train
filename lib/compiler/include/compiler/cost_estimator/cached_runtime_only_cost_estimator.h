#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_CACHED_RUNTIME_ONLY_COST_ESTIMATOR_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_CACHED_RUNTIME_ONLY_COST_ESTIMATOR_H

#include "compiler/cost_estimator/runtime_only_cost_estimator.h"
#include <unordered_map>

namespace FlexFlow {

/**
 * \brief Answers each distinct operator cost query once, and repeats that
 * answer for every later query with the same key.
 *
 * \details A search that re-costs whole graphs asks about the same operator
 * over and over. When the underlying estimator measures kernels, that is both
 * slow and inconsistent: two measurements of one operator differ by noise, so
 * a candidate graph could look better or worse than the graph it came from for
 * no reason but which measurement it got. Caching makes every graph containing
 * an operator pay the same price for it.
 *
 * Tensor movement costs are passed through unchanged.
 */
struct CachedRuntimeOnlyCostEstimator final : public IRuntimeOnlyCostEstimator {
  CachedRuntimeOnlyCostEstimator() = delete;
  explicit CachedRuntimeOnlyCostEstimator(
      RuntimeOnlyCostEstimator const &underlying);

  RuntimeOnlyOpCostMetrics
      estimate_cost(RuntimeOnlyOpCostEstimateKey const &) const override;
  milliseconds_t estimate_cost(TensorSetMovement const &) const override;

private:
  RuntimeOnlyCostEstimator underlying;
  mutable std::unordered_map<RuntimeOnlyOpCostEstimateKey,
                             RuntimeOnlyOpCostMetrics>
      op_costs;
};

RuntimeOnlyCostEstimator
    make_cached_runtime_only_cost_estimator(RuntimeOnlyCostEstimator const &);

} // namespace FlexFlow

#endif
