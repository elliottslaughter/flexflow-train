#include "realm-execution/prepared_invocation.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "utils/containers/transform.h"
#include "utils/containers/try_at.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

std::vector<PreparedInvocation> prepare_invocations(
    DynamicOpenDataflowGraph const &g,
    std::vector<DynamicNodeInvocation> const &execution_order,
    TensorInstanceBacking const &tensor_instance_backing,
    PerDeviceOpStateBacking const &device_state_backing) {

  auto ids_for =
      [&](std::map<DynamicTensorSlot, DynamicValueAttrs> const &slots) {
        return transform(vector_of(values(slots)),
                         [&](DynamicValueAttrs const &value) {
                           return dynamic_graph_get_id_for_value(g, value);
                         });
      };

  return transform(
      execution_order, [&](DynamicNodeInvocation const &invocation) {
        return PreparedInvocation{
            /*invocation=*/invocation,
            /*invocation_id=*/
            dynamic_graph_get_id_for_invocation(g, invocation),
            /*input_ids=*/ids_for(invocation.inputs),
            /*output_ids=*/ids_for(invocation.outputs),
            /*tensor_backing=*/
            subset_tensor_instance_backing_for_invocation(
                tensor_instance_backing, invocation),
            /*device_state=*/try_at(device_state_backing.backing, invocation),
        };
      });
}

} // namespace FlexFlow
