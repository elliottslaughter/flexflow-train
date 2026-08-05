#include "utils/cli/cli_parse.h"
#include "utils/cli/cli_spec.h"
#include "utils/containers/contains.h"
#include "utils/containers/enumerate.h"
#include "utils/containers/generate_map.h"
#include "utils/optional.h"

namespace FlexFlow {

static tl::expected<std::variant<CLIFlagKey, CLINamedArgumentKey>, std::string>
    cli_parse_flag_or_named_argument(CLISpec const &cli,
                                     std::string const &arg) {
  for (auto const &[idx, flag_spec] : enumerate(cli.get_flag_specs())) {
    CLIFlagKey key = CLIFlagKey{idx};
    if (("--" + flag_spec.long_flag) == arg) {
      return key;
    }

    if (flag_spec.short_flag.has_value()) {
      if ((std::string{"-"} + flag_spec.short_flag.value()) == arg) {
        return key;
      }
    }
  }

  for (auto const &[idx, named_argument_spec] :
       enumerate(cli.get_named_argument_specs())) {
    CLINamedArgumentKey key = CLINamedArgumentKey{idx};
    if (("--" + named_argument_spec.long_flag) == arg) {
      return key;
    }
  }

  return tl::unexpected(
      fmt::format("Encountered unknown flag or named argument: {}", arg));
}

tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &cli, std::vector<std::string> const &argv) {
  CLIParseResult result = CLIParseResult{
      /*flags=*/generate_map(cli_get_flag_keys(cli),
                             [](CLIFlagKey const &) -> bool { return false; }),
      /*named_arguments=*/
      generate_map(cli_get_named_argument_keys(cli),
                   [](CLINamedArgumentKey const &)
                       -> std::optional<std::string> { return std::nullopt; }),
      /*positional_arguments=*/{},
  };

  nonnegative_int argv_tok_idx = 0_n;
  auto consume_argv_token = [&]() -> std::optional<std::string> {
    if (argv_tok_idx >= num_elements(argv)) {
      return std::nullopt;
    }

    std::string tok = argv.at(argv_tok_idx.int_from_nonnegative_int());
    argv_tok_idx++;
    return tok;
  };

  auto num_remaining_arg_tokens = [&]() -> nonnegative_int {
    return nonnegative_int{
        argv.size() - argv_tok_idx.int_from_nonnegative_int(),
    };
  };

  nonnegative_int consumed_positional_args = 0_n;
  auto parse_positional_arg =
      [&](std::string const &arg) -> std::optional<std::string> {
    if (consumed_positional_args >=
        cli.get_positional_argument_specs().size()) {
      return fmt::format("Too many positional arguments: expected {}",
                         cli.get_positional_argument_specs().size());
    }

    CLIPositionalArgumentSpec arg_spec = cli.get_positional_argument_specs().at(
        consumed_positional_args.unwrap_nonnegative());

    if (arg_spec.choices.has_value() &&
        !contains(arg_spec.choices.value(), arg)) {
      return fmt::format(
          "Invalid option for positional argument \"{}\": \"{}\"",
          arg_spec.name,
          arg);
    }

    result.positional_arguments.insert(
        {CLIPositionalArgumentKey{consumed_positional_args}, arg});
    consumed_positional_args++;

    return std::nullopt;
  };

  auto parse_flag_or_named_argument =
      [&](std::string const &tok) -> std::optional<std::string> {
    tl::expected<std::variant<CLIFlagKey, CLINamedArgumentKey>, std::string>
        parsed_flag_or_named_argument_result =
            cli_parse_flag_or_named_argument(cli, tok);

    if (!parsed_flag_or_named_argument_result.has_value()) {
      return parsed_flag_or_named_argument_result.error();
    }

    std::variant<CLIFlagKey, CLINamedArgumentKey>
        parsed_flag_or_named_argument =
            parsed_flag_or_named_argument_result.value();

    if (std::holds_alternative<CLIFlagKey>(parsed_flag_or_named_argument)) {
      CLIFlagKey flag_key = std::get<CLIFlagKey>(parsed_flag_or_named_argument);
      result.flags.at(flag_key) = true;
    } else {
      CLINamedArgumentKey named_argument_key =
          std::get<CLINamedArgumentKey>(parsed_flag_or_named_argument);
      CLINamedArgumentSpec arg_spec = cli.at(named_argument_key);

      std::optional<std::string> maybe_value_tok = consume_argv_token();
      if (!maybe_value_tok.has_value()) {
        return fmt::format("Missing value for named argument \"--{}\"",
                           arg_spec.long_flag);
      }

      std::string value_tok = maybe_value_tok.value();
      if (!value_tok.empty() && value_tok.at(0) == '-') {
        return fmt::format("Missing value for named argument \"--{}\"",
                           arg_spec.long_flag);
      }

      if (arg_spec.choices.has_value() &&
          !contains(arg_spec.choices.value(), value_tok)) {
        return fmt::format("Invalid option for named argument \"--{}\": \"{}\"",
                           arg_spec.long_flag,
                           value_tok);
      }

      result.named_arguments.at(named_argument_key) = value_tok;
    }

    return std::nullopt;
  };

  std::string prog_name = assert_unwrap(consume_argv_token());
  while (num_remaining_arg_tokens() > 0) {
    std::string tok = assert_unwrap(consume_argv_token());

    if (!tok.empty() && tok.at(0) == '-') {
      std::optional<std::string> maybe_err_msg =
          parse_flag_or_named_argument(tok);
      if (maybe_err_msg.has_value()) {
        return tl::unexpected(maybe_err_msg.value());
      }
    } else {
      std::optional<std::string> maybe_err_msg = parse_positional_arg(tok);
      if (maybe_err_msg.has_value()) {
        return tl::unexpected(maybe_err_msg.value());
      }
    }
  }

  if (consumed_positional_args != cli.get_positional_argument_specs().size()) {
    return tl::unexpected(
        fmt::format("Not enough positional arguments: found {}, expected {}",
                    consumed_positional_args,
                    cli.get_positional_argument_specs().size()));
  }

  return result;
}

tl::expected<CLIParseResult, std::string>
    cli_parse(CLISpec const &cli, int argc, char const *const *argv) {
  std::vector<std::string> args = {argv, argv + argc};

  return cli_parse(cli, args);
}

} // namespace FlexFlow
