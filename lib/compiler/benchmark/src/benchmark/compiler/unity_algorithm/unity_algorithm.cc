#include "compiler/unity_algorithm/unity_algorithm.h"
#include "compiler/cost_estimator/fake_cost_estimator.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "models/transformer/transformer.h"
#include "models/yolov10/yolov10.h"
#include "pcg/computation_graph.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "substitutions/unlabelled/find_pattern_matches.h"
#include "utils/benchmark_utils.h"
#include "utils/containers/slice.h"

namespace FlexFlow {

void benchmark_unity_algorithm(bool dry_run) {
  // ComputationGraph cg = get_transformer_computation_graph(get_default_transformer_config());
  YOLOv10Config config = get_yolov10_config(
      /*scale=*/YOLOv10Scale::EXTRA_LARGE,
      /*batch_size=*/8_p,
      /*end2end=*/false);
  // config.backbone_config = slice(config.backbone_config, 0, 10);

  ComputationGraph cg = get_yolov10_computation_graph(config);

  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  MachineComputeSpecification full_machine_spec = MachineComputeSpecification{
      /*num_nodes=*/1_p,
      /*num_cpus_per_node=*/1_p,
      /*num_gpus_per_node=*/1_p,
  };

  // Substitution substitution = create_fuse_batch_norm_activation(Activation::SILU, BatchNormMode::SPATIAL);

  std::vector<Substitution> substitution_set =
      get_expanded_substitution_set(full_machine_spec);
  /*
  std::vector<Substitution> substitution_set = {
    create_fuse_batch_norm_activation(Activation::SILU, BatchNormMode::SPATIAL),
  };
  */

  RuntimeOnlyCostEstimator cost_estimator =
      runtime_only_cost_estimator_from_cost_estimator(make_fake_cost_estimator(
          [](OpCostEstimateKey const &k) -> OpCostMetrics {
            return OpCostMetrics{
                /*forward_runtime=*/1.0_ms,
                /*backward_runtime=*/2.0_ms,
                /*memory=*/1_bytes,
            };
          },
          [](TensorSetMovement const &) -> milliseconds_t { return 1.0_ms; }));

  UnitySearchConfig search_config = UnitySearchConfig{
      /*alpha=*/1.0,
      /*budget=*/1000,
      /*max_num_ops=*/1000,
  };

  auto [final_cost, optimized_graph] = her_graph_optimize(
      pcg, cost_estimator, full_machine_spec, search_config, substitution_set);

  std::cout << "final cost: " << final_cost << std::endl;

  // SubParallelComputationGraph subpcg = sub_pcg_from_full_pcg(pcg);

  /*
  std::vector<UnlabelledKwargDataflowGraphPatternMatch> unlabelled_matches =
      find_unlabelled_pattern_matches(get_unlabelled_pattern(substitution.pcg_pattern),
                                      subpcg.raw_graph,
                                      pcg_pattern_criteria(substitution.pcg_pattern, subpcg));
  */

  // std::vector<PCGPatternMatch> pattern_matches = find_pattern_matches(substitution.pcg_pattern, subpcg);
  // LOOP(1, dry_run) {

  // }
  // int x = 0;
  // for (PCGPatternMatch const &match : pattern_matches) {
  //   std::cout << x << "/" << pattern_matches.size() << std::endl;
  //   x++;
  //   subpcg =
  //       apply_substitution(subpcg, substitution, match);
  // }

  /*
  LOOP(1, dry_run) {
    all_pcgs_obtained_by_applying_a_substitution(pcg, substitution_set); 
  }
  */
}

} // namespace FlexFlow
