#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_INTERSECTION_WITH_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_INTERSECTION_WITH_KEYS_H

#include <set>
#include <map>
#include "utils/containers/keys.h"
#include <algorithm>

namespace FlexFlow {

template <typename K, typename V>
std::set<K>
  set_intersection_with_keys(
    std::set<K> const &s,
    std::map<K, V> const &m)
{
  std::set<K> result;

  keys_container<K, V> m_keys{m};

  std::set_intersection(
    s.cbegin(), s.cend(),
    m_keys.cbegin(), m_keys.cend(),
    std::inserter(result, result.begin()));

  return result;
}

} // namespace FlexFlow

#endif
