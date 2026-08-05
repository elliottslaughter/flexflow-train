#ifndef _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_DISCARD_H
#define _FLEXFLOW_LIB_UTILS_TEST_COMMON_INCLUDE_TEST_UTILS_DOCTEST_DISCARD_H

#include <string>

namespace FlexFlow {

template <typename T>
void discard(T const &) {
  return;
};

} // namespace FlexFlow

#endif
