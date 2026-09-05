#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PREPARED_INVOCATION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PREPARED_INVOCATION_H

#include "realm-execution/per_device_op_state_backing.dtg.h"
#include "realm-execution/tensor_instance_backing.dtg.h"
#include "task-spec/dynamic_graph/dynamic_invocation_id_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_id_t.dtg.h"
#include <optional>
#include <vector>

namespace FlexFlow {

/**
 * @brief A \ref DynamicNodeInvocation together with everything about it that
 * does not change from one training iteration to the next.
 *
 * Issuing a task used to redo all of this every iteration: picking the
 * invocation's instances out of the full \ref TensorInstanceBacking, and
 * finding its state in the \ref PerDeviceOpStateBacking. Both are keyed by
 * whole \ref DynamicValueAttrs and \ref DynamicNodeInvocation objects, so every
 * lookup compares parallel tensor shapes, mappings and bidicts, and there are
 * thousands of lookups per iteration. None of the results depend on the
 * iteration, so they are computed once, when the \ref PCGInstance is created.
 */
struct PreparedInvocation {
  DynamicNodeInvocation invocation;

  /**
   * @brief How a launch names this invocation to the node running it.
   *
   * \see register_op_task_args
   */
  dynamic_invocation_id_t invocation_id;

  /**
   * @brief The invocation's inputs and outputs interned against the graph, in
   * the order \c values() yields them.
   *
   * This is what \ref DependencySet is keyed on: an id is a pair of small
   * integers, where the value it stands for is a large structure that is
   * expensive to hash and compare.
   *
   * \warning An id only means anything for the graph it was computed from, so
   * these have to come from the final graph, after every pass has run.
   */
  std::vector<dynamic_value_id_t> input_ids;
  std::vector<dynamic_value_id_t> output_ids;

  /**
   * @brief Just this invocation's instances, rather than every instance in the
   * program.
   */
  TensorInstanceBacking tensor_backing;

  std::optional<DeviceSpecificPtr<PerDeviceOpState>> device_state;
};

/**
 * @brief Precompute the per-iteration-invariant part of each invocation in
 * \p execution_order.
 *
 * \relates PreparedInvocation
 */
std::vector<PreparedInvocation> prepare_invocations(
    DynamicOpenDataflowGraph const &g,
    std::vector<DynamicNodeInvocation> const &execution_order,
    TensorInstanceBacking const &tensor_instance_backing,
    PerDeviceOpStateBacking const &device_state_backing);

} // namespace FlexFlow

#endif
