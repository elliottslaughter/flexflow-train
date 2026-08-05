#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_SPEC_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CLI_CLI_SPEC_H

#include "utils/cli/cli_argument_key.dtg.h"
#include "utils/cli/cli_flag_key.dtg.h"
#include "utils/cli/cli_flag_spec.dtg.h"
#include "utils/cli/cli_named_argument_key.dtg.h"
#include "utils/cli/cli_named_argument_spec.dtg.h"
#include "utils/cli/cli_positional_argument_key.dtg.h"
#include "utils/cli/cli_positional_argument_spec.dtg.h"
#include <vector>

namespace FlexFlow {

struct CLISpec {
public:
  CLISpec();

  explicit CLISpec(
      std::vector<CLIFlagSpec> const &flags,
      std::vector<CLINamedArgumentSpec> const &named_arguments,
      std::vector<CLIPositionalArgumentSpec> const &positional_arguments);

  [[nodiscard]] bool operator==(CLISpec const &) const;
  [[nodiscard]] bool operator!=(CLISpec const &) const;

  [[nodiscard]] bool operator<(CLISpec const &) const;
  [[nodiscard]] bool operator>(CLISpec const &) const;
  [[nodiscard]] bool operator<=(CLISpec const &) const;
  [[nodiscard]] bool operator>=(CLISpec const &) const;

  [[nodiscard]] std::vector<CLIFlagSpec> const &get_flag_specs() const;
  [[nodiscard]] std::vector<CLINamedArgumentSpec> const &
      get_named_argument_specs() const;
  [[nodiscard]] std::vector<CLIPositionalArgumentSpec> const &
      get_positional_argument_specs() const;

  [[nodiscard]] CLIFlagSpec const &at(CLIFlagKey const &) const;
  [[nodiscard]] CLINamedArgumentSpec const &
      at(CLINamedArgumentKey const &) const;
  [[nodiscard]] CLIPositionalArgumentSpec const &
      at(CLIPositionalArgumentKey const &) const;

  [[nodiscard]] CLIArgumentKey add_flag(CLIFlagSpec const &);
  [[nodiscard]] CLIArgumentKey add_named_argument(CLINamedArgumentSpec const &);
  [[nodiscard]] CLIArgumentKey
      add_positional_argument(CLIPositionalArgumentSpec const &);

private:
  std::vector<CLIFlagSpec> flags;
  std::vector<CLINamedArgumentSpec> named_arguments;
  std::vector<CLIPositionalArgumentSpec> positional_arguments;

private:
  void check_invariants() const;

  [[nodiscard]] std::tuple<decltype(flags) const &,
                           decltype(named_arguments) const &,
                           decltype(positional_arguments) const &>
      tie() const;

  friend struct ::std::hash<CLISpec>;
};

std::string format_as(CLISpec const &);
std::ostream &operator<<(std::ostream &, CLISpec const &);

std::vector<CLIFlagKey> cli_get_flag_keys(CLISpec const &);
std::vector<CLINamedArgumentKey> cli_get_named_argument_keys(CLISpec const &);

CLIArgumentKey cli_add_help_flag(CLISpec &);

} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::CLISpec> {
  size_t operator()(::FlexFlow::CLISpec const &) const;
};

} // namespace std

namespace nlohmann {

template <>
struct adl_serializer<::FlexFlow::CLISpec> {
  static ::FlexFlow::CLISpec from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::CLISpec const &t);
};

} // namespace nlohmann

#endif
