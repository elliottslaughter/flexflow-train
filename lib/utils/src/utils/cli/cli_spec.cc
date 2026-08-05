#include "utils/cli/cli_spec.h"
#include "utils/containers/are_all_distinct.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/count.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/transform.h"
#include "utils/fmt/vector.h"
#include "utils/hash-utils.h"
#include "utils/hash/tuple.h"
#include "utils/hash/vector.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

CLISpec::CLISpec(
    std::vector<CLIFlagSpec> const &raw_flags,
    std::vector<CLINamedArgumentSpec> const &raw_named_arguments,
    std::vector<CLIPositionalArgumentSpec> const &raw_positional_arguments)
    : flags(raw_flags), named_arguments(raw_named_arguments),
      positional_arguments(raw_positional_arguments) {
  this->check_invariants();
}

CLISpec::CLISpec() : flags({}), named_arguments({}), positional_arguments({}) {}

bool CLISpec::operator==(CLISpec const &other) const {
  return this->tie() == other.tie();
}

bool CLISpec::operator!=(CLISpec const &other) const {
  return this->tie() != other.tie();
}

bool CLISpec::operator<(CLISpec const &other) const {
  return this->tie() < other.tie();
}

bool CLISpec::operator>(CLISpec const &other) const {
  return this->tie() > other.tie();
}

bool CLISpec::operator<=(CLISpec const &other) const {
  return this->tie() <= other.tie();
}

bool CLISpec::operator>=(CLISpec const &other) const {
  return this->tie() >= other.tie();
}

std::vector<CLIFlagSpec> const &CLISpec::get_flag_specs() const {
  return this->flags;
}

std::vector<CLINamedArgumentSpec> const &
    CLISpec::get_named_argument_specs() const {
  return this->named_arguments;
}

std::vector<CLIPositionalArgumentSpec> const &
    CLISpec::get_positional_argument_specs() const {
  return this->positional_arguments;
}

CLIFlagSpec const &CLISpec::at(CLIFlagKey const &k) const {
  return this->flags.at(k.raw_idx.int_from_nonnegative_int());
}

CLINamedArgumentSpec const &CLISpec::at(CLINamedArgumentKey const &k) const {
  return this->named_arguments.at(k.raw_idx.int_from_nonnegative_int());
}

CLIPositionalArgumentSpec const &
    CLISpec::at(CLIPositionalArgumentKey const &k) const {
  return this->positional_arguments.at(k.raw_idx.int_from_nonnegative_int());
}

CLIArgumentKey CLISpec::add_flag(CLIFlagSpec const &flag_spec) {
  CLIArgumentKey key = CLIArgumentKey{CLIFlagKey{num_elements(this->flags)}};
  this->flags.push_back(flag_spec);
  this->check_invariants();
  return key;
}

CLIArgumentKey CLISpec::add_named_argument(CLINamedArgumentSpec const &arg) {
  CLIArgumentKey key =
      CLIArgumentKey{CLINamedArgumentKey{num_elements(this->named_arguments)}};
  this->named_arguments.push_back(arg);
  this->check_invariants();
  return key;
}

CLIArgumentKey
    CLISpec::add_positional_argument(CLIPositionalArgumentSpec const &arg) {
  CLIArgumentKey key = CLIArgumentKey{
      CLIPositionalArgumentKey{num_elements(this->positional_arguments)}};
  this->positional_arguments.push_back(arg);
  this->check_invariants();
  return key;
}

void CLISpec::check_invariants() const {
  {
    std::vector<std::string> flag_long_flag_names = transform(
        this->flags, [&](CLIFlagSpec const &flag_spec) -> std::string {
          return flag_spec.long_flag;
        });

    std::vector<std::string> named_argument_long_flag_names = transform(
        this->named_arguments,
        [&](CLINamedArgumentSpec const &named_argument_spec) -> std::string {
          return named_argument_spec.long_flag;
        });

    std::vector<std::string> all_long_flag_names =
        concat_vectors(flag_long_flag_names, named_argument_long_flag_names);

    ASSERT(are_all_distinct(all_long_flag_names));
  }

  {
    std::vector<char> short_flag_names = filtrans(
        this->flags, [&](CLIFlagSpec const &flag_spec) -> std::optional<char> {
          return flag_spec.short_flag;
        });

    ASSERT(are_all_distinct(short_flag_names));
  }

  {
    std::vector<std::string> positional_argument_names =
        transform(this->positional_arguments,
                  [&](CLIPositionalArgumentSpec const &positional_argument_spec)
                      -> std::string { return positional_argument_spec.name; });

    ASSERT(are_all_distinct(positional_argument_names));
  }
}

std::tuple<std::vector<CLIFlagSpec> const &,
           std::vector<CLINamedArgumentSpec> const &,
           std::vector<CLIPositionalArgumentSpec> const &>
    CLISpec::tie() const {
  return std::tie(
      this->flags, this->named_arguments, this->positional_arguments);
}

std::string format_as(CLISpec const &cli) {
  return fmt::format(
      "<CLISpec flags={} named_arguments={} positional_arguments={}>",
      cli.get_flag_specs(),
      cli.get_named_argument_specs(),
      cli.get_positional_argument_specs());
}

std::ostream &operator<<(std::ostream &s, CLISpec const &cli) {
  return (s << fmt::to_string(cli));
}

std::vector<CLIFlagKey> cli_get_flag_keys(CLISpec const &cli) {
  return transform(nonnegative_range(num_elements(cli.get_flag_specs())),
                   [](nonnegative_int idx) { return CLIFlagKey{idx}; });
}

std::vector<CLINamedArgumentKey>
    cli_get_named_argument_keys(CLISpec const &cli) {
  return transform(
      nonnegative_range(num_elements(cli.get_named_argument_specs())),
      [](nonnegative_int idx) { return CLINamedArgumentKey{idx}; });
}

CLINamedArgumentSpec
    cli_get_named_argument_spec(CLISpec const &cli,
                                CLINamedArgumentKey const &key) {
  return cli.get_named_argument_specs().at(
      key.raw_idx.int_from_nonnegative_int());
}

CLIArgumentKey cli_add_help_flag(CLISpec &cli) {
  CLIFlagSpec help_flag =
      CLIFlagSpec{"help", 'h', "show this help message and exit"};
  return cli.add_flag(help_flag);
}

} // namespace FlexFlow

namespace std {

size_t
    hash<::FlexFlow::CLISpec>::operator()(::FlexFlow::CLISpec const &x) const {
  return ::FlexFlow::get_std_hash(x.tie());
}

} // namespace std

namespace nlohmann {

::FlexFlow::CLISpec
    adl_serializer<::FlexFlow::CLISpec>::from_json(json const &j) {
  return ::FlexFlow::CLISpec{
      /*flag_specs=*/j.at("flag_specs")
          .template get<std::vector<::FlexFlow::CLIFlagSpec>>(),
      /*named_argument_specs=*/
      j.at("named_argument_specs")
          .template get<std::vector<::FlexFlow::CLINamedArgumentSpec>>(),
      /*positional_argument_specs=*/
      j.at("positional_argument_specs")
          .template get<std::vector<::FlexFlow::CLIPositionalArgumentSpec>>(),
  };
}

void adl_serializer<::FlexFlow::CLISpec>::to_json(
    json &j, ::FlexFlow::CLISpec const &cli) {
  j["flag_specs"] = cli.get_flag_specs();
  j["named_argument_specs"] = cli.get_named_argument_specs();
  j["positional_argument_specs"] = cli.get_positional_argument_specs();
}

} // namespace nlohmann
