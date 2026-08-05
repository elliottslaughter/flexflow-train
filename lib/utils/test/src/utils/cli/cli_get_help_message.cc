#include "utils/cli/cli_get_help_message.h"
#include "utils/join_strings.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("cli_get_help_message(string, CLISpec)") {
    std::string program_name = "prog_name";

    SUBCASE("no flags or positional arguments") {
      CLISpec cli = CLISpec{
          /*flags=*/{},
          /*named_arguments=*/{},
          /*positional_arguments=*/{},
      };

      std::string result = cli_get_help_message(program_name, cli);
      std::string correct = ("usage: prog_name\n");

      CHECK(result == correct);
    }

    SUBCASE("only flags") {
      CLISpec cli = CLISpec{
          /*flags=*/{
              CLIFlagSpec{
                  /*long_flag=*/"flag-1",
                  /*short_flat=*/'f',
                  /*description=*/std::nullopt,
              },
          },
          /*named_arguments=*/{},
          /*positional_arguments=*/{},
      };

      std::string result = cli_get_help_message(program_name, cli);
      std::string correct = ("usage: prog_name [-f]\n"
                             "\n"
                             "options:\n"
                             "  -f, --flag-1\n");

      CHECK(result == correct);
    }

    SUBCASE("only named arguments") {
      CLISpec cli = CLISpec{
          /*flags=*/{},
          /*named_arguments=*/
          {
              CLINamedArgumentSpec{
                  /*long_flag=*/"named-arg",
                  /*metavar=*/"NAMED_ARG",
                  /*choices=*/std::nullopt,
                  /*description=*/std::nullopt,
              },
          },
          /*positional_arguments=*/{},
      };

      std::string result = cli_get_help_message(program_name, cli);
      std::string correct = ("usage: prog_name [--named-arg NAMED_ARG]\n"
                             "\n"
                             "options:\n"
                             "  --named-arg NAMED_ARG\n");

      CHECK(result == correct);
    }

    SUBCASE("only positional arguments") {
      CLISpec cli = CLISpec{
          /*flags=*/{},
          /*named_arguments=*/{},
          /*positional_arguments=*/
          {
              CLIPositionalArgumentSpec{
                  /*name=*/"pos-arg-1",
                  /*choices=*/std::nullopt,
                  /*description=*/std::nullopt,
              },
          },
      };

      std::string result = cli_get_help_message(program_name, cli);
      std::string correct = ("usage: prog_name pos-arg-1\n"
                             "\n"
                             "positional arguments:\n"
                             "  pos-arg-1\n");

      CHECK(result == correct);
    }

    SUBCASE("flag formatting") {
      SUBCASE("flag with shortname") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    "flag",
                    'f',
                    std::nullopt,
                },
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name [-f]\n"
                               "\n"
                               "options:\n"
                               "  -f, --flag\n");

        CHECK(result == correct);
      }

      SUBCASE("flag without shortname") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    "flag",
                    std::nullopt,
                    std::nullopt,
                },
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name [--flag]\n"
                               "\n"
                               "options:\n"
                               "  --flag\n");

        CHECK(result == correct);
      }

      SUBCASE("flags are displayed in provided order") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    "flag2",
                    std::nullopt,
                    std::nullopt,
                },
                CLIFlagSpec{
                    "flag1",
                    std::nullopt,
                    std::nullopt,
                },
            },
            /*named_arguments=*/{},
            /*positional_arguments=*/{},
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name [--flag2] [--flag1]\n"
                               "\n"
                               "options:\n"
                               "  --flag2\n"
                               "  --flag1\n");

        CHECK(result == correct);
      }
    }

    SUBCASE("named argument formatting") {
      SUBCASE("without choices") {
        CLISpec cli = CLISpec{
            /*flags=*/{},
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"named-arg",
                    /*metavar=*/"NAMED_ARG",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*positional_arguments=*/{},
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name [--named-arg NAMED_ARG]\n"
                               "\n"
                               "options:\n"
                               "  --named-arg NAMED_ARG\n");

        CHECK(result == correct);
      }

      SUBCASE("with choices") {
        SUBCASE("choices are not empty") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  CLINamedArgumentSpec{
                      /*long_flag=*/"named-arg",
                      /*metavar=*/"NAMED_ARG",
                      /*choices=*/
                      std::vector<std::string>{
                          "one",
                          "two",
                          "three",
                      },
                      /*description=*/std::nullopt,
                  },
              },
              /*positional_arguments=*/{},
          };

          std::string result = cli_get_help_message(program_name, cli);
          std::string correct =
              ("usage: prog_name [--named-arg {one,two,three}]\n"
               "\n"
               "options:\n"
               "  --named-arg {one,two,three}\n");

          CHECK(result == correct);
        }

        SUBCASE("choices are empty") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/
              {
                  CLINamedArgumentSpec{
                      /*long_flag=*/"named-arg",
                      /*metavar=*/"NAMED_ARG",
                      /*choices=*/std::vector<std::string>{},
                      /*description=*/std::nullopt,
                  },
              },
              /*positional_arguments=*/{},
          };

          std::string result = cli_get_help_message(program_name, cli);
          std::string correct = ("usage: prog_name [--named-arg {}]\n"
                                 "\n"
                                 "options:\n"
                                 "  --named-arg {}\n");

          CHECK(result == correct);
        }
      }

      SUBCASE("are displayed in provided order") {
        CLISpec cli = CLISpec{
            /*flags=*/{},
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"named-arg-2",
                    /*metavar=*/"NAMED_ARG_2",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
                CLINamedArgumentSpec{
                    /*long_flag=*/"named-arg-1",
                    /*metavar=*/"NAMED_ARG_1",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
            /*positional_arguments=*/{},
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name [--named-arg-2 NAMED_ARG_2] "
                               "[--named-arg-1 NAMED_ARG_1]\n"
                               "\n"
                               "options:\n"
                               "  --named-arg-2 NAMED_ARG_2\n"
                               "  --named-arg-1 NAMED_ARG_1\n");

        CHECK(result == correct);
      }
    }

    SUBCASE("positional argument formatting") {
      SUBCASE("without choices") {
        CLISpec cli = CLISpec{
            /*flags=*/{},
            /*named_arguments=*/{},
            /*positional_arguments=*/
            {
                CLIPositionalArgumentSpec{
                    "posarg",
                    std::nullopt,
                    std::nullopt,
                },
            },
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name posarg\n"
                               "\n"
                               "positional arguments:\n"
                               "  posarg\n");

        CHECK(result == correct);
      }

      SUBCASE("with choices") {
        SUBCASE("choices are not empty") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  CLIPositionalArgumentSpec{
                      "posarg",
                      std::vector<std::string>{"red", "blue", "green"},
                      std::nullopt,
                  },
              },
          };

          std::string result = cli_get_help_message(program_name, cli);
          std::string correct = ("usage: prog_name {red,blue,green}\n"
                                 "\n"
                                 "positional arguments:\n"
                                 "  {red,blue,green}\n");

          CHECK(result == correct);
        }

        SUBCASE("choices are empty") {
          CLISpec cli = CLISpec{
              /*flags=*/{},
              /*named_arguments=*/{},
              /*positional_arguments=*/
              {
                  CLIPositionalArgumentSpec{
                      "posarg",
                      std::vector<std::string>{},
                      std::nullopt,
                  },
              },
          };

          std::string result = cli_get_help_message(program_name, cli);
          std::string correct = ("usage: prog_name {}\n"
                                 "\n"
                                 "positional arguments:\n"
                                 "  {}\n");

          CHECK(result == correct);
        }
      }

      SUBCASE("are displayed in provided order") {
        CLISpec cli = CLISpec{
            /*flags=*/{},
            /*named_arguments=*/{},
            /*positional_arguments=*/
            {
                CLIPositionalArgumentSpec{
                    /*name=*/"posarg2",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
                CLIPositionalArgumentSpec{
                    /*name=*/"posarg1",
                    /*choices=*/std::nullopt,
                    /*description=*/std::nullopt,
                },
            },
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct = ("usage: prog_name posarg2 posarg1\n"
                               "\n"
                               "positional arguments:\n"
                               "  posarg2\n"
                               "  posarg1\n");

        CHECK(result == correct);
      }
    }

    SUBCASE("flag and argument alignment") {
      SUBCASE("flags are longer") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    /*long_flag=*/"flag1",
                    /*short_flag=*/'1',
                    /*description=*/"flag1 description",
                },
                CLIFlagSpec{
                    /*long_flag=*/"flag2-is-long",
                    /*short_flag=*/std::nullopt,
                    /*descriptiopn=*/"flag2-is-long description",
                },
            },
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg",
                    /*metavar=*/"N",
                    /*choices=*/std::nullopt,
                    /*description=*/"help text for namedarg",
                },
            },
            /*positional_arguments=*/
            {
                CLIPositionalArgumentSpec{
                    /*name=*/"posarg",
                    /*choices=*/std::nullopt,
                    /*description=*/"help text for posarg",
                },
            },
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct =
            ("usage: prog_name [-1] [--flag2-is-long] [--namedarg N] posarg\n"
             "\n"
             "positional arguments:\n"
             "  posarg           help text for posarg\n"
             "\n"
             "options:\n"
             "  -1, --flag1      flag1 description\n"
             "  --flag2-is-long  flag2-is-long description\n"
             "  --namedarg N     help text for namedarg\n");

        CHECK(result == correct);
      }

      SUBCASE("named args are longer") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    /*long_flag=*/"flag1",
                    /*short_flag=*/'1',
                    /*description=*/"flag1 description",
                },
            },
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg",
                    /*metavar=*/"N",
                    /*choices=*/std::nullopt,
                    /*description=*/"help text for namedarg",
                },
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg-is-long",
                    /*metavar=*/"L",
                    /*choices=*/std::nullopt,
                    /*description=*/"help text for long arg",
                },
            },
            /*positional_arguments=*/
            {
                CLIPositionalArgumentSpec{
                    /*name=*/"posarg",
                    /*choices=*/std::nullopt,
                    /*description=*/"help text for posarg",
                },
            },
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct =
            ("usage: prog_name [-1] [--namedarg N] [--namedarg-is-long L] "
             "posarg\n"
             "\n"
             "positional arguments:\n"
             "  posarg                help text for posarg\n"
             "\n"
             "options:\n"
             "  -1, --flag1           flag1 description\n"
             "  --namedarg N          help text for namedarg\n"
             "  --namedarg-is-long L  help text for long arg\n");

        CHECK(result == correct);
      }

      SUBCASE("pos args are longer") {
        CLISpec cli = CLISpec{
            /*flags=*/{
                CLIFlagSpec{
                    "flag1",
                    '1',
                    "flag1 description",
                },
            },
            /*named_arguments=*/
            {
                CLINamedArgumentSpec{
                    /*long_flag=*/"namedarg",
                    /*metavar=*/"N",
                    /*choices=*/std::nullopt,
                    /*description=*/"help text for namedarg",
                },
            },
            /*positional_arguments=*/
            {
                CLIPositionalArgumentSpec{
                    "posarg1-is-very-long",
                    std::nullopt,
                    "help text for posarg1-is-very-long",
                },
                CLIPositionalArgumentSpec{
                    "posarg2",
                    std::nullopt,
                    "help text for posarg2",
                },
            },
        };

        std::string result = cli_get_help_message(program_name, cli);
        std::string correct =
            ("usage: prog_name [-1] [--namedarg N] posarg1-is-very-long "
             "posarg2\n"
             "\n"
             "positional arguments:\n"
             "  posarg1-is-very-long  help text for posarg1-is-very-long\n"
             "  posarg2               help text for posarg2\n"
             "\n"
             "options:\n"
             "  -1, --flag1           flag1 description\n"
             "  --namedarg N          help text for namedarg\n");

        CHECK(result == correct);
      }

      SUBCASE("line break behavior") {
        SUBCASE("line breaks max out other argument alignments") {
          CLISpec cli = CLISpec{
              /*flags=*/{
                  CLIFlagSpec{
                      "flag",
                      'f',
                      "flag help text",
                  },
              },
              /*named_arguments=*/
              {
                  CLINamedArgumentSpec{
                      /*long_flag=*/"namedarg",
                      /*metavar=*/"N",
                      /*choices=*/std::nullopt,
                      /*description=*/"namedarg help text",
                  },
              },
              /*positional_arguments=*/
              {
                  CLIPositionalArgumentSpec{
                      "abcdefghijklmnopqrstuvwxyz0123456789",
                      std::nullopt,
                      "long arg help text",
                  },
                  CLIPositionalArgumentSpec{
                      "posarg",
                      std::nullopt,
                      "posarg help text",
                  },
              },
          };

          std::string result = cli_get_help_message(program_name, cli);
          std::string correct =
              ("usage: prog_name [-f] [--namedarg N] "
               "abcdefghijklmnopqrstuvwxyz0123456789 posarg\n"
               "\n"
               "positional arguments:\n"
               "  abcdefghijklmnopqrstuvwxyz0123456789\n"
               "                        long arg help text\n"
               "  posarg                posarg help text\n"
               "\n"
               "options:\n"
               "  -f, --flag            flag help text\n"
               "  --namedarg N          namedarg help text\n");

          CHECK(result == correct);
        }

        SUBCASE("positional argument line break behavior") {
          SUBCASE("positional arguments cause a line break at or above "
                  "formatted-length 22") {
            std::string arg_name = "aaaaaaaaaaaaaaaaaaaaaa";
            REQUIRE(arg_name.size() == 22);

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    CLIPositionalArgumentSpec{
                        arg_name,
                        std::nullopt,
                        "help text",
                    },
                },
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name aaaaaaaaaaaaaaaaaaaaaa\n"
                                   "\n"
                                   "positional arguments:\n"
                                   "  aaaaaaaaaaaaaaaaaaaaaa\n"
                                   "                        help text\n");

            CHECK(result == correct);
          }

          SUBCASE("positional arguments do not cause a line break below "
                  "formatted-length 22") {
            std::string arg_name = "aaaaaaaaaaaaaaaaaaaaa";
            REQUIRE(arg_name.size() == 21);

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    CLIPositionalArgumentSpec{
                        arg_name,
                        std::nullopt,
                        "help text",
                    },
                },
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name aaaaaaaaaaaaaaaaaaaaa\n"
                                   "\n"
                                   "positional arguments:\n"
                                   "  aaaaaaaaaaaaaaaaaaaaa\n"
                                   "                        help text\n");
          }
        }

        SUBCASE("named argument line break behavior") {
          SUBCASE("named arguments cause a line break at or above "
                  "formatted-length 21") {
            std::string metavar = "BBB";
            std::string arg_name = "bbbbbbbbbbbbbbb";

            {
              std::string formatted = "--" + arg_name + " " + metavar;
              REQUIRE(formatted.size() == 21);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/
                {
                    CLINamedArgumentSpec{
                        /*long_flag=*/arg_name,
                        /*metavar=*/metavar,
                        /*choices=*/std::nullopt,
                        /*description=*/"arg description",
                    },
                },
                /*positional_arguments=*/{},
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name [--bbbbbbbbbbbbbbb BBB]\n"
                                   "\n"
                                   "options:\n"
                                   "  --bbbbbbbbbbbbbbb BBB\n"
                                   "                        arg description\n");

            CHECK(result == correct);
          }

          SUBCASE("named arguments do not cause a line break below "
                  "formatted-length 21") {
            std::string metavar = "BBB";
            std::string arg_name = "bbbbbbbbbbbbbb";

            {
              std::string formatted = "--" + arg_name + " " + metavar;
              REQUIRE(formatted.size() == 20);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/
                {
                    CLINamedArgumentSpec{
                        /*long_flag=*/arg_name,
                        /*metavar=*/metavar,
                        /*choices=*/std::nullopt,
                        /*description=*/"arg description",
                    },
                },
                /*positional_arguments=*/{},
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name [--bbbbbbbbbbbbbb BBB]\n"
                                   "\n"
                                   "options:\n"
                                   "  --bbbbbbbbbbbbbb BBB  arg description\n");

            CHECK(result == correct);
          }
        }

        SUBCASE("flag line break behavior") {
          SUBCASE("flags cause a line break at or above formatted-length 21") {
            std::string arg_name = "bbbbbbbbbbbbbbb";
            {
              std::string formatted = "-b, --" + arg_name;
              REQUIRE(formatted.size() == 21);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{
                    CLIFlagSpec{
                        arg_name,
                        'b',
                        "flag description",
                    },
                },
                /*named_arguments=*/{},
                /*positional_arguments=*/{},
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct =
                ("usage: prog_name [-b]\n"
                 "\n"
                 "options:\n"
                 "  -b, --bbbbbbbbbbbbbbb\n"
                 "                        flag description\n");

            CHECK(result == correct);
          }

          SUBCASE("flags do not cause a line break below formatted-length 21") {
            std::string arg_name = "bbbbbbbbbbbbbb";
            {
              std::string formatted = "-b, --" + arg_name;
              REQUIRE(formatted.size() == 20);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{
                    CLIFlagSpec{
                        arg_name,
                        'b',
                        "flag description",
                    },
                },
                /*named_arguments=*/{},
                /*positional_arguments=*/{},
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct =
                ("usage: prog_name [-b]\n"
                 "\n"
                 "options:\n"
                 "  -b, --bbbbbbbbbbbbbb  flag description\n");

            CHECK(result == correct);
          }
        }

        SUBCASE("positional argument choice line breakpoint formatting") {
          SUBCASE("positional argument choices cause a line break at or above "
                  "formatted-length 21") {
            std::vector<std::string> choices = {
                "a", "b", "c", "d", "e", "fffffffff"};
            {
              std::string formatted_choices =
                  "{" + join_strings(choices, ",") + "}";
              REQUIRE(formatted_choices.size() == 21);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    CLIPositionalArgumentSpec{
                        "posarg",
                        choices,
                        "help text",
                    },
                },
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name {a,b,c,d,e,fffffffff}\n"
                                   "\n"
                                   "positional arguments:\n"
                                   "  {a,b,c,d,e,fffffffff}\n"
                                   "                        help text\n");

            CHECK(result == correct);
          }

          SUBCASE("positional argument choices do not cause a line break below "
                  "formatted-length 21") {
            std::vector<std::string> choices = {
                "a", "b", "c", "d", "e", "ffffffff"};
            {
              std::string formatted_choices =
                  "{" + join_strings(choices, ",") + "}";
              REQUIRE(formatted_choices.size() == 20);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    CLIPositionalArgumentSpec{
                        "posarg",
                        choices,
                        "help text",
                    },
                },
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name {a,b,c,d,e,ffffffff}\n"
                                   "\n"
                                   "positional arguments:\n"
                                   "  {a,b,c,d,e,ffffffff}  help text\n");

            CHECK(result == correct);
          }
        }

        SUBCASE("named argument choice line breakpoint formatting") {
          SUBCASE("named argument choices cause a line break at or above "
                  "formatted-length 21") {
            std::vector<std::string> choices = {
                "a", "b", "c", "d", "e", "fffffffff"};
            {
              std::string formatted_choices =
                  "{" + join_strings(choices, ",") + "}";
              REQUIRE(formatted_choices.size() == 21);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    CLIPositionalArgumentSpec{
                        "posarg",
                        choices,
                        "help text",
                    },
                },
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name {a,b,c,d,e,fffffffff}\n"
                                   "\n"
                                   "positional arguments:\n"
                                   "  {a,b,c,d,e,fffffffff}\n"
                                   "                        help text\n");

            CHECK(result == correct);
          }

          SUBCASE("named argument choices do not cause a line break below "
                  "formatted-length 21") {
            std::vector<std::string> choices = {
                "a", "b", "c", "d", "e", "ffffffff"};
            {
              std::string formatted_choices =
                  "{" + join_strings(choices, ",") + "}";
              REQUIRE(formatted_choices.size() == 20);
            }

            CLISpec cli = CLISpec{
                /*flags=*/{},
                /*named_arguments=*/{},
                /*positional_arguments=*/
                {
                    CLIPositionalArgumentSpec{
                        "posarg",
                        choices,
                        "help text",
                    },
                },
            };

            std::string result = cli_get_help_message(program_name, cli);
            std::string correct = ("usage: prog_name {a,b,c,d,e,ffffffff}\n"
                                   "\n"
                                   "positional arguments:\n"
                                   "  {a,b,c,d,e,ffffffff}  help text\n");

            CHECK(result == correct);
          }
        }
      }
    }
  }
}
