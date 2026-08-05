#include "utils/cli/cli_get_help_message.h"
#include "utils/containers/concat_vectors.h"
#include "utils/containers/maximum.h"
#include "utils/containers/transform.h"
#include "utils/fmt/vector.h"
#include "utils/integer_conversions.h"
#include "utils/join_strings.h"
#include <sstream>

namespace FlexFlow {

std::string cli_get_help_message(std::string const &program_name,
                                 CLISpec const &cli) {
  auto render_pos_arg = [](CLIPositionalArgumentSpec const &pos_arg_spec) {
    if (pos_arg_spec.choices.has_value()) {
      return "{" + join_strings(pos_arg_spec.choices.value(), ",") + "}";
    } else {
      return pos_arg_spec.name;
    }
  };

  auto render_named_arg = [](CLINamedArgumentSpec const &named_argument_spec) {
    std::ostringstream oss;
    oss << "--" << named_argument_spec.long_flag << " ";
    if (named_argument_spec.choices.has_value()) {
      oss << ("{" + join_strings(named_argument_spec.choices.value(), ",") +
              "}");
    } else {
      oss << named_argument_spec.metavar;
    }
    return oss.str();
  };

  auto render_flag_option_column_key = [](CLIFlagSpec const &flag_spec) {
    std::ostringstream oss;
    if (flag_spec.short_flag.has_value()) {
      oss << "-" << flag_spec.short_flag.value() << ", ";
    }
    oss << "--" << flag_spec.long_flag;
    return oss.str();
  };

  std::ostringstream oss;

  oss << "usage: " << program_name;
  for (CLIFlagSpec const &flag_spec : cli.get_flag_specs()) {
    if (flag_spec.short_flag.has_value()) {
      oss << " [-" << flag_spec.short_flag.value() << "]";
    } else {
      oss << " [--" << flag_spec.long_flag << "]";
    }
  }
  for (CLINamedArgumentSpec const &named_argument_spec :
       cli.get_named_argument_specs()) {
    oss << " [" << render_named_arg(named_argument_spec) << "]";
  }
  for (CLIPositionalArgumentSpec const &pos_arg_spec :
       cli.get_positional_argument_specs()) {
    oss << " " << render_pos_arg(pos_arg_spec);
  }

  oss << std::endl;

  std::vector<std::string> all_arg_columns = concat_vectors(std::vector{
      transform(cli.get_positional_argument_specs(), render_pos_arg),
      transform(cli.get_named_argument_specs(), render_named_arg),
      transform(cli.get_flag_specs(), render_flag_option_column_key),
  });
  std::vector<size_t> all_arg_column_widths =
      transform(all_arg_columns, [](std::string const &s) { return s.size(); });

  if (!all_arg_columns.empty()) {
    int max_column_width =
        std::min(int_from_size_t(maximum(all_arg_column_widths)), 20);

    auto render_column = [&](std::string const &key,
                             std::optional<std::string> const &description) {
      if (description.has_value()) {
        if (key.size() > max_column_width) {
          return "  " + key + "\n" + std::string(24, ' ') + description.value();
        } else {
        }
        return fmt::format(
            "  {:<{}}  {}", key, max_column_width, description.value());
      } else {
        return fmt::format("  {}", key);
      }
    };

    if (!cli.get_positional_argument_specs().empty()) {
      oss << std::endl;
      oss << "positional arguments:" << std::endl;

      if (!cli.get_positional_argument_specs().empty()) {
        for (CLIPositionalArgumentSpec const &pos_arg_spec :
             cli.get_positional_argument_specs()) {
          oss << render_column(render_pos_arg(pos_arg_spec),
                               pos_arg_spec.description)
              << std::endl;
        }
      }
    }

    if (!(cli.get_flag_specs().empty() &&
          cli.get_named_argument_specs().empty())) {
      oss << std::endl;
      oss << "options:" << std::endl;

      for (CLIFlagSpec const &flag_spec : cli.get_flag_specs()) {
        oss << render_column(render_flag_option_column_key(flag_spec),
                             flag_spec.description)
            << std::endl;
      }
      for (CLINamedArgumentSpec const &named_argument_spec :
           cli.get_named_argument_specs()) {
        oss << render_column(render_named_arg(named_argument_spec),
                             named_argument_spec.description)
            << std::endl;
      }
    }
  }

  return oss.str();
}

} // namespace FlexFlow
