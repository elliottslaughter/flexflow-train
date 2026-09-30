#include <doctest/doctest.h>
#include "utils/containers/set_intersection_with_keys.h"
#include <set>
#include "test/utils/doctest/fmt/set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("set_intersection_with_keys") {
    std::set<int> s = {0, 2, 3, 5};
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};

    std::set<int> result = set_intersection_with_keys(s, m);
    std::set<int> correct = {2, 3};

    CHECK(result == correct);
  }
}
