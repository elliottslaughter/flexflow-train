#include "utils/deduplicated_queue.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct deduplicated_queue<T>;

} // namespace FlexFlow
