#include "pcg/file_format/v1/v1_computation_graph.h"
#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/cli/cli_get_help_message.h"
#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_parse_result.h"
#include "utils/cli/cli_spec.h"
#include <fstream>

using namespace FlexFlow;

int main(int argc, char **argv) {
  CLISpec cli = empty_cli_spec();

  CLIArgumentKey arg_key_help = cli_add_help_flag(cli);

  CLIArgumentKey key_cg_json = cli_add_positional_argument(
      cli,
      CLIPositionalArgumentSpec{
          "cg_json",
          std::nullopt,
          "path to a file containing computation graph, encoded as JSON"});

  CLIArgumentKey key_mpcg_json_output = cli_add_positional_argument(
      cli,
      CLIPositionalArgumentSpec{
          "mpcg_json_output",
          std::nullopt,
          "path to write the resulting mapping PCG, encoded as JSON"});

  std::vector<std::string> strategy_options = {"passthrough",
                                            "data_parallel",
                                            "unity",
                                            "mcmc"};
  CLIArgumentKey key_strategy = cli_add_positional_argument(
      cli,
      CLIPositionalArgumentSpec{
          "strategy", strategy_options, "compilation strategy for building the mapped PCG"});

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

  std::string cg_json = cli_get_argument(parsed, key_cg_json);
  std::string mpcg_json_output = cli_get_argument(parsed, key_mpcg_json_output);
  std::string strategy = cli_get_argument(parsed, key_strategy);

  ComputationGraph cg = [&]() {
    std::ifstream f{cg_json};
    nlohmann::json cg_json = nlohmann::json::parse(f);
    return from_v1(cg_json.get<V1ComputationGraph>());
  }();

  MappedParallelComputationGraph mpcg = [&]() {
    if (strategy == "passthrough") {
      ParallelComputationGraph pcg = pcg_from_computation_graph(cg);
      std::map<parallel_layer_guid_t, MappedOperatorTaskGroup>
          mapped_op_task_groups;
      return mapped_pcg_from_pcg_and_mapped_op_task_groups(
          pcg,
          mapped_op_task_groups);
    } else if (strategy == "data_parallel") {
      NOT_IMPLEMENTED();
    } else if (strategy == "unity") {
      NOT_IMPLEMENTED();
    } else if (strategy == "mcmc") {
      NOT_IMPLEMENTED();
    } else {
      PANIC("no such strategy: {}", strategy);
    }
  }();

  {
    std::ofstream f{mpcg_json_output};
    nlohmann::json mpcg_json = to_v1(mpcg);
    f << mpcg_json;
  }

  return 0;
}
