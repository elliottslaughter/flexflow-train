#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_KEYS_H

#include <map>
#include <set>

namespace FlexFlow {

template <typename K, typename V>
struct keys_container {
  explicit keys_container(std::map<K, V> const &m)
    : m(m)
  { }

  struct iterator {
  private:
    using It = typename std::map<K, V>::const_iterator;
  public:
    using difference_type = long;
    using value_type = K;
    using pointer = K const *;
    using reference = K const &;
    using iterator_category = std::input_iterator_tag;

  public:
    explicit iterator(It it)
      : it(it)
    {}

    iterator &operator++() {
      it++;
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }

    bool operator==(iterator other) const {
      return this->it == other.it;
    }

    bool operator!=(iterator other) const {
      return this->it != other.it;
    }

    reference operator*() const {
      return this->it->first;
    }
  private:
    It it;
  };

  using const_iterator = iterator;
  using value_type = typename iterator::value_type;
  using difference_type = typename iterator::difference_type;
  using pointer = typename iterator::pointer;
  using reference = typename iterator::reference;
  using const_reference = typename iterator::reference;

  iterator begin() const {
    return iterator(m.cbegin());
  }

  iterator end() const {
    return iterator(m.cend());
  }

  const_iterator cbegin() const {
    return this->begin();
  }

  const_iterator cend() const {
    return this->end();
  }

private:
  std::map<K, V> const &m;
};

template <typename K, typename V>
std::set<K> keys(std::map<K, V> const &c) {
  keys_container<K, V> ks{c};
  return std::set<K>{ks.cbegin(), ks.cend()};
}

template <typename K, typename V>
keys_container<K, V> lazy_keys(std::map<K, V> const &c) {
  keys_container<K, V> ks{c};
  return ks;
}

} // namespace FlexFlow

#endif
