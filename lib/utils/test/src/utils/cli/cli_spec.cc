#include "utils/cli/cli_spec.h"
#include "test/utils/doctest/discard.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("CLISpec") {
    SUBCASE("CLISpec::CLISpec(std::vector<>, ...)") {
      SUBCASE("throws if there are duplicate long flags in flags") {
        CHECK_THROWS(CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    /*long_flag=*/"flag1",
                    /*short_flag=*/'f',
                    /*description=*/std::nullopt,
                },
                CLIFlagSpec{
                    /*long_flag=*/"flag1",
                    /*short_flag=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        });
      }

      SUBCASE("throws if there are duplicate short flags in flags") {
        CHECK_THROWS(CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    /*long_flag=*/"flag1",
                    /*short_flag=*/'f',
                    /*description=*/std::nullopt,
                },
                CLIFlagSpec{
                    /*long_flag=*/"flag2",
                    /*short_flag=*/'f',
                    /*description=*/std::nullopt,
                },
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        });
      }

      SUBCASE("throws if there are duplicate long flags in named arguments") {
        CHECK_THROWS(CLISpec{
            /*flags=*/{},
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    "namedarg1",
                    /*metavar=*/"A",
                    /*choices=*/std::vector<std::string>{"one", "two", "three"},
                    /*description=*/std::nullopt,
                },
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg1",
                    /*metavar=*/"B",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*positional_arguments=*/{},
        });
      }

      SUBCASE("throws if there are duplicate long flags between flags and "
              "named arguments") {
        CHECK_THROWS(CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    "flag1",
                    std::nullopt,
                    std::nullopt,
                },
            },
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"flag1",
                    /*metavar=*/"A",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*positional_arguments=*/{},
        });
      }
    }

    CLIFlagSpec flag1_spec = CLIFlagSpec{
        /*long_flag=*/"flag1",
        /*short_flag=*/'f',
        /*description=*/std::nullopt,
    };
    CLIFlagSpec flag2_spec = CLIFlagSpec{
        /*long_flag=*/"flag2",
        /*short_flag=*/std::nullopt,
        /*description=*/std::nullopt,
    };
    CLIFlagSpec flag3_spec = CLIFlagSpec{
        /*long_flag=*/"flag3",
        /*short_flag=*/'a',
        /*description=*/std::nullopt,
    };

    CLINamedArgumentSpec namedarg1_spec = CLINamedArgumentSpec{
        /*long_flag=*/"namedarg1",
        /*metavar=*/"A",
        /*choices=*/std::vector<std::string>{"one", "two", "three"},
        /*description=*/std::nullopt,
    };
    CLINamedArgumentSpec namedarg2_spec = CLINamedArgumentSpec{
        /*long_flag=*/"namedarg2",
        /*metavar=*/"B",
        /*choices=*/std::nullopt,
        /*description=*/std::nullopt,
    };
    CLIPositionalArgumentSpec posarg1_spec = CLIPositionalArgumentSpec{
        /*name=*/"posarg1",
        /*choices=*/std::vector<std::string>{"red", "blue", "green"},
        /*description=*/std::nullopt,
    };
    CLIPositionalArgumentSpec posarg2_spec = CLIPositionalArgumentSpec{
        /*name=*/"posarg2",
        /*choices=*/std::nullopt,
        /*descriptiopn=*/std::nullopt,
    };

    CLISpec cli = CLISpec{
        /*flags=*/{
            flag1_spec,
            flag2_spec,
            flag3_spec,
        },
        /*named_arguments=*/
        {
            namedarg1_spec,
            namedarg2_spec,
        },
        /*positional_arguments=*/
        {
            posarg1_spec,
            posarg2_spec,
        },
    };

    SUBCASE("add_flag") {
      SUBCASE("throws if flag already exists with that short_flag") {
        CLIFlagSpec input = CLIFlagSpec{
            /*long_flag=*/"flag4",
            /*short_flag=*/flag1_spec.short_flag,
            /*description=*/std::nullopt,
        };

        CHECK_THROWS(discard(cli.add_flag(input)));
      }

      SUBCASE("throws if flag already exists with that long_flag") {
        CLIFlagSpec input = CLIFlagSpec{
            /*long_flag=*/flag3_spec.long_flag,
            /*short_flag=*/std::nullopt,
            /*description=*/std::nullopt,
        };

        CHECK_THROWS(discard(cli.add_flag(input)));
      }

      SUBCASE("throws if identical flag already exists") {
        CLIFlagSpec input = flag3_spec;

        CHECK_THROWS(discard(cli.add_flag(input)));
      }

      SUBCASE("throws if named argument already exists with that long_flag") {
        CLIFlagSpec input = CLIFlagSpec{
            /*long_flag=*/namedarg1_spec.long_flag,
            /*short_flag=*/std::nullopt,
            /*description=*/std::nullopt,
        };

        CHECK_THROWS(discard(cli.add_flag(input)));
      }

      SUBCASE("correct usage") {
        CLIFlagSpec input = CLIFlagSpec{
            /*long_flag=*/"flag4",
            /*short_flag=*/'4',
            /*description=*/std::nullopt,
        };

        CLIArgumentKey result_key = cli.add_flag(input);

        SUBCASE("returns the right key") {
          CLIArgumentKey correct = CLIArgumentKey{CLIFlagKey{3_n}};

          CHECK(result_key == correct);
        }

        SUBCASE("adds the new flag spec") {
          CLISpec correct = CLISpec{
              /*flags=*/{
                  flag1_spec,
                  flag2_spec,
                  flag3_spec,
                  input,
              },
              /*named_arguments=*/
              {
                  namedarg1_spec,
                  namedarg2_spec,
              },
              /*positional_arguments=*/
              {
                  posarg1_spec,
                  posarg2_spec,
              },
          };

          CHECK(cli == correct);
        }
      }
    }

    SUBCASE("add_named_argument") {
      SUBCASE("throws if flag already exists with that long_flag") {
        CLINamedArgumentSpec input = CLINamedArgumentSpec{
            /*long_flag=*/flag1_spec.long_flag,
            /*metavar=*/"INPUT",
            /*short_flag=*/std::nullopt,
            /*description=*/"my new arg",
        };

        CHECK_THROWS(discard(cli.add_named_argument(input)));
      }

      SUBCASE("throws if named argument already exists with that long_flag") {
        CLINamedArgumentSpec input = CLINamedArgumentSpec{
            /*long_flag=*/namedarg1_spec.long_flag,
            /*metavar=*/"INPUT",
            /*short_flag=*/std::nullopt,
            /*description=*/"my new arg",
        };

        CHECK_THROWS(discard(cli.add_named_argument(input)));
      }

      SUBCASE("throws if identical named arg already exists") {
        CLINamedArgumentSpec input = namedarg2_spec;

        CHECK_THROWS(discard(cli.add_named_argument(input)));
      }

      SUBCASE("correct usage") {
        CLINamedArgumentSpec input = CLINamedArgumentSpec{
            /*long_flag=*/"namedarg3",
            /*metavar=*/"NAMEDARG3",
            /*choices=*/std::nullopt,
            /*description=*/std::nullopt,
        };

        CLIArgumentKey result_key = cli.add_named_argument(input);

        SUBCASE("returns the right key") {
          CLIArgumentKey correct = CLIArgumentKey{CLINamedArgumentKey{2_n}};

          CHECK(result_key == correct);
        }

        SUBCASE("adds the new flag spec") {
          CLISpec correct = CLISpec{
              /*flags=*/{
                  flag1_spec,
                  flag2_spec,
                  flag3_spec,
              },
              /*named_arguments=*/
              {
                  namedarg1_spec,
                  namedarg2_spec,
                  input,
              },
              /*positional_arguments=*/
              {
                  posarg1_spec,
                  posarg2_spec,
              },
          };

          CHECK(cli == correct);
        }
      }
    }

    SUBCASE("add_positional_argument") {
      SUBCASE("throws if arg already exists with that name") {
        CLIPositionalArgumentSpec input = CLIPositionalArgumentSpec{
            /*name=*/posarg1_spec.name,
            /*choices=*/std::nullopt,
            /*description=*/"my new arg",
        };

        CHECK_THROWS(discard(cli.add_positional_argument(input)));
      }

      SUBCASE("throws if identical positional arg already exists") {
        CLIPositionalArgumentSpec input = posarg2_spec;

        CHECK_THROWS(discard(cli.add_positional_argument(input)));
      }

      SUBCASE("correct usage") {
        CLIPositionalArgumentSpec input = CLIPositionalArgumentSpec{
            /*name=*/"posarg3",
            /*choices=*/std::nullopt,
            /*description=*/"my new arg",
        };

        CLIArgumentKey result_key = cli.add_positional_argument(input);

        SUBCASE("returns the right key") {
          CLIArgumentKey correct =
              CLIArgumentKey{CLIPositionalArgumentKey{2_n}};

          CHECK(result_key == correct);
        }

        SUBCASE("adds the new flag spec") {
          CLISpec correct = CLISpec{
              /*flags=*/{
                  flag1_spec,
                  flag2_spec,
                  flag3_spec,
              },
              /*named_arguments=*/
              {
                  namedarg1_spec,
                  namedarg2_spec,
              },
              /*positional_arguments=*/
              {
                  posarg1_spec,
                  posarg2_spec,
                  input,
              },
          };

          CHECK(cli == correct);
        }
      }
    }
  }
}
