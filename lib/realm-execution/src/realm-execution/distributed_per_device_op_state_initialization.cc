#include "realm-execution/distributed_per_device_op_state_initialization.h"
#include "local-execution/per_device_op_state_initialization.h"
#include "realm-execution/tasks/impl/per_device_op_state_init_task.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "realm-execution/tensor_instance_backing.h"
#include "task-spec/dynamic_graph/dynamic_node_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "utils/containers/map_values.h"
#include "utils/containers/maybe_get_only.h"
#include "utils/containers/set_of.h"
#include "utils/containers/values.h"
#include "utils/optional.h"
#include <map>
#include <vector>
#include <optional>
#include <utility>

namespace FlexFlow {

PerDeviceOpStateBacking perform_distributed_per_device_op_state_initialization(
    RealmContext &ctx,
    DynamicOpenDataflowGraph const &dg,
    TensorInstanceBacking const &tensor_instance_backing,
    ProfilingSettings const &profiling_settings,
    DistributedFfHandle const &device_handle,
    OptimizerAttrs const &optimizer_attrs,
    Realm::Event precondition) {

  // Initialize all operators and save the per-device op state
  ASSERT(no_nodes_are_initialized(dg));

  // The forward and backward invocations of an operator differ only in their
  // task type (see pass_expand_node), and some operators save state during the
  // forward pass that the backward pass then reads back (batch norm's saved
  // mean and inverse variance, dropout's RNG state, cuDNN attention's reserve
  // space). Those invocations therefore have to share a single per-device op
  // state rather than each initializing their own, so group the invocations by
  // everything but their task type and initialize each group once.
  auto state_sharing_key =
      [](DynamicNodeInvocation const &invocation) -> DynamicNodeAttrs {
    DynamicNodeAttrs key = invocation.node_attrs;
    key.task_type = std::nullopt;
    return key;
  };

  std::map<DynamicNodeAttrs, std::vector<DynamicNodeInvocation>> groups;
  for (DynamicNodeInvocation const &invocation : dg.invocations) {
    // Nodes mapped to multiple devices are always parallel operators and don't
    // have any initialization to perform anyway
    std::optional<global_device_id_t> device_id =
        maybe_get_only(assert_unwrap(invocation.node_attrs.device_ids));
    if (!device_id.has_value()) {
      continue;
    }

    groups[state_sharing_key(invocation)].push_back(invocation);
  }

  std::map<DynamicNodeInvocation, DeviceSpecificPtr<PerDeviceOpState> *>
      device_state_map;
  for (auto const &[key, group] : groups) {
    // Initialize from the forward invocation, whose bindings are the ones the
    // init kernels expect.
    DynamicNodeInvocation representative = [&] {
      for (DynamicNodeInvocation const &invocation : group) {
        if (invocation.node_attrs.task_type == DynamicTaskType::FWD) {
          return invocation;
        }
      }
      return group.at(0);
    }();

    std::optional<global_device_id_t> device_id =
        maybe_get_only(assert_unwrap(representative.node_attrs.device_ids));

    Realm::Processor target_proc =
        ctx.processor_from_global_device_id(assert_unwrap(device_id));

    TensorInstanceBacking tensor_backing =
        subset_tensor_instance_backing_for_invocation(tensor_instance_backing,
                                                      representative);

    DeviceSpecificPtr<PerDeviceOpState> *device_state_ptr =
        new DeviceSpecificPtr<PerDeviceOpState>{
            ctx.get_current_global_device_id(), std::nullopt};

    std::optional<Realm::Event> completion_event =
        spawn_per_device_op_state_init_task(ctx,
                                            target_proc,
                                            representative,
                                            tensor_backing,
                                            profiling_settings,
                                            device_handle.at(target_proc),
                                            optimizer_attrs,
                                            device_state_ptr,
                                            precondition);

    if (completion_event.has_value()) {
      for (DynamicNodeInvocation const &invocation : group) {
        device_state_map.insert(std::pair{invocation, device_state_ptr});
      }
    } else {
      // Task doesn't require initialization, clean up and don't store result
      delete device_state_ptr;
    }
  }

  ctx.get_outstanding_events().wait();

  auto deref = [](DeviceSpecificPtr<PerDeviceOpState> *const &p) { return *p; };
  std::map<DynamicNodeInvocation, DeviceSpecificPtr<PerDeviceOpState>> result =
      map_values(device_state_map, deref);

  // Several invocations can share one pointer, so deduplicate before deleting.
  for (DeviceSpecificPtr<PerDeviceOpState> *device_state_ptr :
       set_of(values(device_state_map))) {
    delete device_state_ptr;
  }

  return PerDeviceOpStateBacking{/*backing=*/result};
}

} // namespace FlexFlow
