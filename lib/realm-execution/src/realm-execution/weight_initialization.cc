#include "realm-execution/weight_initialization.h"
#include "realm-execution/tasks/impl/weight_init_task.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/parallel_tensor_mapping.h"
#include "task-spec/dynamic_graph/weight_initialization.h"
#include "utils/optional.h"
#include <functional>

namespace FlexFlow {

void perform_weight_initialization(
    RealmContext &ctx,
    DynamicOpenDataflowGraph const &g,
    TensorInstanceBacking const &tensor_instance_backing,
    Realm::Event precondition) {

  for (auto const &[value, initializer] : get_weight_initializers(g)) {
    // Every default initializer op-attrs picks carries a seed of 0, so without
    // something to tell one weight from another every layer of a given shape
    // would come out with bit-identical contents. This is computed here rather
    // than in the task so that every shard of a weight agrees on it.
    size_t salt = std::hash<dynamic_tensor_guid_t>{}(value.tensor_guid);

    global_device_id_t device_id = pt_mapping_get_device_for_coord(
        assert_unwrap(value.mapping), assert_unwrap(value.shard_coord));
    Realm::Processor target_proc =
        ctx.processor_from_global_device_id(device_id);

    TensorInstanceBacking shard_backing = TensorInstanceBacking{
        /*backing=*/{{value, tensor_instance_backing.backing.at(value)}},
    };

    spawn_weight_init_task(/*ctx=*/ctx,
                           /*target_proc=*/target_proc,
                           /*value=*/value,
                           /*tensor_backing=*/shard_backing,
                           /*initializer=*/initializer,
                           /*salt=*/salt,
                           /*precondition=*/precondition);
  }
}

} // namespace FlexFlow
