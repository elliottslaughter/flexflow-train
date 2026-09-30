#include "compiler/unity_algorithm/unity_algorithm.h"
#include "models/yolov10/yolov10.h"
#include "pcg/pcg_from_computation_graph.h"
#include "substitutions/unity_substitution_set.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/unlabelled/find_pattern_matches.h"
#include "models/transformer/transformer.h"
#include "utils/containers/slice.h"
#include "pcg/computation_graph.h"
#include "substitutions/apply_substitution/apply_substitution.h"
#include "compiler/cost_estimator/runtime_only_cost_estimator_from_cost_estimator.h"
#include "compiler/cost_estimator/fake_cost_estimator.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse_result.h"

using namespace FlexFlow;

int main(int argc, char **argv) {
  CLISpec cli;

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_dot = cli.add_flag(CLIFlagSpec{
      /*long_flag=*/"dot",
      /*short_flag=*/std::nullopt,
      /*description=*/"print the optimized graph in dot format",
  });

  ASSERT(argc >= 1);
  std::string prog_name = argv[0];

  CLIParseResult parsed = ({
    tl::expected<CLIParseResult, std::string> result =
        cli_parse(cli, argc, argv);
    if (!result.has_value()) {
      std::string error_msg = result.error();
      std::cerr << cli_get_help_message(prog_name, cli);
      std::cerr << std::endl;
      std::cerr << "error: " << error_msg << std::endl;
      return 1;
    }

    result.value();
  });

  bool help = cli_get_flag(parsed, arg_key_help);
  if (help) {
    std::cerr << cli_get_help_message(prog_name, cli);
    return 1;
  }

  bool dot = cli_get_flag(parsed, key_dot);

  YOLOv10Config config =
      get_yolov10_config(
        /*scale=*/YOLOv10Scale::EXTRA_LARGE,
        /*batch_size=*/8_p,
        /*end2end=*/false);

  ComputationGraph cg =
    get_yolov10_computation_graph(config);

  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);

  MachineComputeSpecification full_machine_spec = MachineComputeSpecification{
      /*num_nodes=*/1_p,
      /*num_cpus_per_node=*/1_p,
      /*num_gpus_per_node=*/1_p,
  };

  std::vector<Substitution> substitution_set = get_expanded_substitution_set(full_machine_spec);

  RuntimeOnlyCostEstimator cost_estimator =
      runtime_only_cost_estimator_from_cost_estimator(
          make_fake_cost_estimator(
              [](OpCostEstimateKey const &k) -> OpCostMetrics {
                return OpCostMetrics{
                    /*forward_runtime=*/1.0_ms,
                    /*backward_runtime=*/2.0_ms,
                    /*memory=*/1_bytes,
                };
              },
              [](TensorSetMovement const &) -> milliseconds_t {
                return 1.0_ms;
              }));

  UnitySearchConfig search_config = UnitySearchConfig{
      /*alpha=*/1.0,
      /*budget=*/1000,
      /*max_num_ops=*/1000,
  };

  auto new_best_hook = [](int iter, 
                          int num_remaining_in_queue, 
                          milliseconds_t old_best, 
                          milliseconds_t new_best) 
  {
    std::cerr 
      << fmt::format("Found new best ({} -> {})", old_best, new_best)
      << std::endl;
  };

  auto [final_cost, optimized_graph] = her_graph_optimize(
      /*pcg=*/pcg,
      /*cost_estimator=*/cost_estimator,
      /*resources=*/full_machine_spec,
      /*search_config=*/search_config,
      /*substitutions=*/substitution_set,
      /*new_candidates_hook=*/std::nullopt,
      /*start_candidate_hook=*/std::nullopt,
      /*finished_candidate_hook=*/std::nullopt,
      /*new_best_hook=*/new_best_hook);

  if (dot) {
    std::cout << pcg_as_dot(optimized_graph) << std::endl;
  } else {
    fmt::println("Final cost: {}", final_cost);
  }

  return 0;
}
