#include "compiler/cost_estimator/cached_runtime_only_cost_estimator.h"
#include "compiler/machine_mapping/machine_view.h"
#include "internal/runtime_only_cost_estimator_for_test.h"
#include "op-attrs/parallel_tensor_shape.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_cached_runtime_only_cost_estimator") {
    int num_measurements = 0;

    // Answers differently every time, as measuring a kernel does.
    RuntimeOnlyCostEstimator measuring = make_fake_runtime_only_cost_estimator(
        [&](RuntimeOnlyOpCostEstimateKey const &) {
          num_measurements++;
          return RuntimeOnlyOpCostMetrics{
              /*forward_runtime=*/milliseconds_t{float(num_measurements)},
              /*backward_runtime=*/milliseconds_t{float(num_measurements)},
          };
        },
        [](TensorSetMovement const &) { return 0_ms; });

    RuntimeOnlyCostEstimator cached =
        make_cached_runtime_only_cost_estimator(measuring);

    ParallelTensorShape shape = lift_to_parallel(TensorShape{
        TensorDims{FFOrdered{4_p, 8_p}},
        DataType::FLOAT,
    });

    auto key_for = [&](OperatorType op_type) {
      return RuntimeOnlyOpCostEstimateKey{
          /*op_attrs=*/PCGOperatorAttrs{ElementUnaryAttrs{
              /*op_type=*/op_type,
              /*scalar=*/std::nullopt,
          }},
          /*input_shapes=*/{{TensorSlotName::INPUT, shape}},
          /*weight_shapes=*/{},
          /*output_shapes=*/{{TensorSlotName::OUTPUT, shape}},
          /*machine_view=*/
          make_single_device_machine_view(MachineSpaceCoordinate{0_n, 0_n}),
      };
    };

    RuntimeOnlyOpCostEstimateKey relu = key_for(OperatorType::RELU);
    RuntimeOnlyOpCostEstimateKey sigmoid = key_for(OperatorType::SIGMOID);

    RuntimeOnlyOpCostMetrics first = cached.estimate_cost(relu);
    RuntimeOnlyOpCostMetrics again = cached.estimate_cost(relu);
    RuntimeOnlyOpCostMetrics other = cached.estimate_cost(sigmoid);

    CHECK(again == first);
    CHECK(other != first);
    CHECK(num_measurements == 2);
  }
}
