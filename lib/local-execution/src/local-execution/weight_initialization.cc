#include "local-execution/weight_initialization.h"
#include "kernels/initializer_kernels.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/weight_initialization.h"
#include "utils/optional.h"
#include <functional>

namespace FlexFlow {

void perform_weight_initialization(DynamicOpenDataflowGraph const &g) {
  for (auto const &[value, initializer] : get_weight_initializers(g)) {
    // Every default initializer op-attrs picks carries a seed of 0, so without
    // something to tell one weight from another every layer of a given shape
    // would come out with bit-identical contents.
    size_t salt = std::hash<dynamic_tensor_guid_t>{}(value.tensor_guid);

    initialize_tensor(
        assert_unwrap(value.accessor).require_write(), initializer, salt);
  }
}

} // namespace FlexFlow
