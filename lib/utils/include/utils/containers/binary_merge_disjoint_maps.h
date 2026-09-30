#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_MAPS_H

#include <libassert/assert.hpp>
#include "utils/containers/try_merge_nondisjoint_maps.h"
#include "utils/optional.h"

namespace FlexFlow {

template <typename K, typename V>
std::map<K, V> binary_merge_disjoint_maps(std::map<K, V> lhs,
                                          std::map<K, V> const &rhs) {
  return assert_unwrap(try_merge_nondisjoint_maps(lhs, rhs));
}

} // namespace FlexFlow

#endif
