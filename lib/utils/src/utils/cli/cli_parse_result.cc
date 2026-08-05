#include "utils/cli/cli_parse_result.h"

namespace FlexFlow {

bool cli_get_flag(CLIParseResult const &result, CLIArgumentKey const &key) {
  return result.flags.at(key.get<CLIFlagKey>());
}

std::string cli_get_positional_argument(CLIParseResult const &result,
                                        CLIArgumentKey const &key) {
  return result.positional_arguments.at(key.get<CLIPositionalArgumentKey>());
}

std::optional<std::string> cli_get_named_argument(CLIParseResult const &result,
                                                  CLIArgumentKey const &key) {
  return result.named_arguments.at(key.get<CLINamedArgumentKey>());
}

} // namespace FlexFlow
