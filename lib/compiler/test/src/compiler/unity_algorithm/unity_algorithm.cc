#include "compiler/unity_algorithm/unity_algorithm.h"
#include "compiler/cost_estimator/fake_cost_estimator.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "models/yolov10/yolov10.h"
#include "models/yolov10/yolov10_scale.dtg.h"
#include "op-attrs/parallel_tensor_dims.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/replica_type.dtg.h"
#include "op-attrs/shard_parallel_dim.h"
#include "pcg/computation_graph_builder.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_builder.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "utils/integer_conversions.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  // TEST_CASE("graph_optimize") {
  //   ComputationGraph cg = [&] {
  //     ComputationGraphBuilder b;
  //     TensorShape input_tensor_shape = TensorShape{
  //         TensorDims{
  //             FFOrdered<positive_int>{
  //                 32_p,
  //                 64_p,
  //             },
  //         },
  //         DataType::FLOAT,
  //     };
  //     tensor_guid_t t = b.create_input(input_tensor_shape, CreateGrad::YES);
  //     t = b.dense(t,
  //                 /*outDim=*/16_p,
  //                 /*activation=*/std::nullopt);
  //     t = b.gelu(t);
  //     t = b.dense(t,
  //                 /*outDim=*/12_p,
  //                 /*activation=*/std::nullopt,
  //                 /*use_bias=*/false,
  //                 /*data_type=*/DataType::FLOAT,
  //                 /*kernel_initializer=*/std::nullopt,
  //                 /*bias_initializer=*/std::nullopt);
  //     t = b.relu(t);
  //     t = b.dense(t,
  //                 /*outDim=*/8_p,
  //                 /*activation=*/Activation::RELU);
  //     return b.computation_graph;
  //   }();

  //   ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  //   RuntimeOnlyCostEstimator cost_estimator =
  //       runtime_only_cost_estimator_from_cost_estimator(
  //           make_fake_cost_estimator(
  //               [](OpCostEstimateKey const &k) -> OpCostMetrics {
  //                 return OpCostMetrics{
  //                     /*forward_runtime=*/1.0_ms,
  //                     /*backward_runtime=*/2.0_ms,
  //                     /*memory=*/1_bytes,
  //                 };
  //               },
  //               [](TensorSetMovement const &) -> milliseconds_t {
  //                 return 1.0_ms;
  //               }));

  //   MachineComputeSpecification full_machine_spec = MachineComputeSpecification{
  //       /*num_nodes=*/2_p,
  //       /*num_cpus_per_node=*/1_p,
  //       /*num_gpus_per_node=*/1_p,
  //   };

  //   std::vector<Substitution> substitution_set = get_unity_substitution_set(full_machine_spec);

  //   SUBCASE("do not apply substitution") {
  //     UnitySearchConfig search_config = UnitySearchConfig{
  //         /*alpha=*/1.0,
  //         /*budget=*/0,
  //         /*max_num_ops=*/100,
  //     };
  //     SearchResult result =
  //         graph_optimize(pcg, cost_estimator, full_machine_spec, search_config, substitution_set);
  //     CHECK(pcgs_are_isomorphic(pcg, result.pcg));
  //   }

  //   SUBCASE("apply substitution") {
  //     UnitySearchConfig search_config = UnitySearchConfig{
  //         /*alpha=*/1.0,
  //         /*budget=*/1,
  //         /*max_num_ops=*/100,
  //     };
  //     SearchResult result =
  //         graph_optimize(pcg, cost_estimator, full_machine_spec, search_config, substitution_set);
  //   }
  // }

  TEST_CASE("graph_optimize (batch norm + activation fusion)") {
    ParallelComputationGraph pcg = [] {
      ParallelComputationGraphBuilder b;
      parallel_tensor_guid_t t = b.create_input_tensor(TensorShape{
          TensorDims{FFOrdered{2_p, 3_p, 4_p, 4_p}},
          DataType::FLOAT,
      });
      t = b.batch_norm(t,
                       /*affine=*/true,
                       /*activation=*/std::nullopt,
                       /*eps=*/1e-5,
                       /*momentum=*/0.1,
                       /*mode=*/BatchNormMode::SPATIAL);
      t = b.relu(t);
      return b.pcg;
    }();

    MachineComputeSpecification one_gpu = MachineComputeSpecification{
        /*num_nodes=*/1_p,
        /*num_cpus_per_node=*/1_p,
        /*num_gpus_per_node=*/1_p,
    };

    UnitySearchConfig search_config = UnitySearchConfig{
        /*alpha=*/1.0,
        /*budget=*/1000,
        /*max_num_ops=*/1000,
    };

    auto estimator_charging = [](milliseconds_t fused_batch_norm_cost) {
      return runtime_only_cost_estimator_from_cost_estimator(
          make_fake_cost_estimator(
              [=](OpCostEstimateKey const &k) -> OpCostMetrics {
                milliseconds_t cost = [&] {
                  if (k.op_attrs.has<BatchNormAttrs>()) {
                    return k.op_attrs.get<BatchNormAttrs>()
                                   .activation.has_value()
                               ? fused_batch_norm_cost
                               : 2_ms;
                  } else if (k.op_attrs.has<ElementUnaryAttrs>()) {
                    return 1_ms;
                  } else {
                    return 0_ms;
                  }
                }();
                return OpCostMetrics{
                    /*forward_runtime=*/cost,
                    /*backward_runtime=*/cost,
                    /*memory=*/0_bytes,
                };
              },
              [](TensorSetMovement const &) { return 0_ms; }));
    };

    auto batch_norm_activations = [](ParallelComputationGraph const &g) {
      std::vector<std::optional<Activation>> result;
      for (parallel_layer_guid_t l : pcg_get_parallel_layers(g)) {
        PCGOperatorAttrs attrs = get_parallel_layer_attrs(g, l).op_attrs;
        if (attrs.has<BatchNormAttrs>()) {
          result.push_back(attrs.get<BatchNormAttrs>().activation);
        }
      }
      return result;
    };

    SUBCASE("fuses when the fused kernel is cheaper") {
      SearchResult result =
          graph_optimize(pcg,
                         estimator_charging(2.5_ms),
                         one_gpu,
                         search_config,
                         get_expanded_substitution_set(one_gpu));

      CHECK(batch_norm_activations(result.pcg) ==
            std::vector<std::optional<Activation>>{Activation::RELU});
      CHECK(pcg_get_parallel_layers(result.pcg).size() ==
            pcg_get_parallel_layers(pcg).size() - 1);
      CHECK(result.machine_mapping.machine_views.size() ==
            pcg_get_parallel_layers(result.pcg).size());
    }

    SUBCASE("leaves the graph alone when the fused kernel is not cheaper") {
      SearchResult result =
          graph_optimize(pcg,
                         estimator_charging(3.5_ms),
                         one_gpu,
                         search_config,
                         get_expanded_substitution_set(one_gpu));

      CHECK(batch_norm_activations(result.pcg) ==
            std::vector<std::optional<Activation>>{std::nullopt});
      CHECK(pcgs_are_isomorphic(result.pcg, pcg));
    }
  }
}
