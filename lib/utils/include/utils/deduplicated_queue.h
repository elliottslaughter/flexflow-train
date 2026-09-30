#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_DEDUPLICATED_QUEUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_DEDUPLICATED_QUEUE_H

#include "utils/containers/contains.h"
#include <queue>
#include <unordered_set>

namespace FlexFlow {

template <typename Elem>
struct deduplicated_queue {
public:
  Elem const &front() const {
    return impl.front();
  }

  bool empty() const {
    return impl.empty();
  }

  size_t size() const {
    return impl.size();
  }

  void push(Elem const &e) {
    if (!contains(seen, e)) {
      impl.push(e);
      seen.insert(e);
    }
  }

  void pop() {
    seen.erase(impl.front());
    impl.pop();
  }
private:
  std::queue<Elem> impl;
  std::unordered_set<Elem> seen;
};

} // namespace FlexFlow

#endif
