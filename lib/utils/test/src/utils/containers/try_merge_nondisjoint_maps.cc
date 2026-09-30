#include <doctest/doctest.h>
#include "utils/containers/try_merge_nondisjoint_maps.h"
#include <string>
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/optional.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("try_merge_nondisjoint_maps") {
    SUBCASE("inputs are disjoint") {
      std::map<int, std::string> lhs = {{1, "one"}, {2, "two"}};
      std::map<int, std::string> rhs = {{3, "three"}, {4, "four"}};

      std::optional<std::map<int, std::string>> result = try_merge_nondisjoint_maps(lhs, rhs);
      std::optional<std::map<int, std::string>> correct =
        std::map<int, std::string>{
          {1, "one"},
          {2, "two"},
          {3, "three"},
          {4, "four"},
        };

      CHECK(result == correct);
    }

    SUBCASE("inputs are not disjoint, but agree") {
      std::map<int, std::string> lhs = {{1, "one"}, {2, "two"}};
      std::map<int, std::string> rhs = {{2, "two"}, {3, "three"}};

      std::optional<std::map<int, std::string>> result = try_merge_nondisjoint_maps(lhs, rhs);
      std::optional<std::map<int, std::string>> correct = 
        std::map<int, std::string>{
          {1, "one"},
          {2, "two"},
          {3, "three"},
        };

      CHECK(result == correct);
    }

    SUBCASE("inputs are not disjoint and disagree") {
      std::map<int, std::string> lhs = {{1, "ONE"}, {2, "TWO"}};
      std::map<int, std::string> rhs = {{2, "two"}, {3, "three"}};

      std::optional<std::map<int, std::string>> result = try_merge_nondisjoint_maps(lhs, rhs);
      std::optional<std::map<int, std::string>> correct = std::nullopt;

      CHECK(result == correct);
    }
  }
}


