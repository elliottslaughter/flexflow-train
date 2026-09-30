#include "utils/containers/set_intersection_with_keys.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<0>;

template
  std::set<K>
    set_intersection_with_keys(
      std::set<K> const &,
      std::map<K, V> const &);

} // namespace FlexFlow
