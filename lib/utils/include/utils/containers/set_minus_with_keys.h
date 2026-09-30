#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_MINUX_WITH_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SET_MINUX_WITH_KEYS_H

namespace FlexFlow {

template <typename K, typename V>
std::set<K>
  set_minus_with_keys(
    std::set<K> const &s,
    std::map<K, V> const &m)
{
  std::set<K> result;

  keys_container<K, V> m_keys{m};

  std::set_difference(
    s.cbegin(), s.cend(),
    m_keys.cbegin(), m_keys.cend(),
    std::inserter(result, result.begin()));

  return result;
}

template <typename K, typename V>
std::set<K>
  set_minus_with_keys(
    std::map<K, V> const &m,
    std::set<K> const &s)
{
  std::set<K> result;

  keys_container<K, V> m_keys{m};

  std::set_difference(
    m_keys.cbegin(), m_keys.cend(),
    s.cbegin(), s.cend(),
    std::inserter(result, result.begin()));

  return result;
}

} // namespace FlexFlow

#endif
