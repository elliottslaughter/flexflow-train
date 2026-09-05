#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INVOCATION_FUSION_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_INVOCATION_FUSION_H

#include "pcg/optimizer_attrs.dtg.h"
#include "realm-execution/prepared_invocation.h"
#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/global_device_id_t.dtg.h"
#include <optional>
#include <vector>

namespace FlexFlow {

/**
 * @brief A run of \ref PreparedInvocation that is issued as a single Realm
 * operation.
 *
 * A group of one is just an invocation issued the way it always was; a larger
 * group is one Realm task whose body runs its members' task bodies back to
 * back. See \ref group_invocations_for_fusion for when that is allowed.
 */
struct InvocationGroup {
  /**
   * @brief The group's members, in topological order.
   *
   * A fused body runs them in exactly this order, so the order is what makes
   * the members' dependencies on one another hold.
   */
  std::vector<PreparedInvocation> members;

  /**
   * @brief The union of the members' inputs and outputs, deduplicated.
   *
   * The group depends on, and is depended on for, everything any of its
   * members touches. Dependencies between members are carried by the order
   * they run in rather than by \ref DependencySet, so they need no events.
   */
  std::vector<dynamic_value_id_t> input_ids;
  std::vector<dynamic_value_id_t> output_ids;
};

/**
 * @brief How a fused group is named to the node that runs it.
 *
 * A group is identified by its first member, which is unique to it because
 * groups do not overlap.
 *
 * \relates InvocationGroup
 */
dynamic_invocation_id_t
    get_group_id_for_invocation_group(InvocationGroup const &);

bool is_fused_invocation_group(InvocationGroup const &);

/**
 * @brief The \ref DynamicTaskType every member of the group runs, which the
 * passes select on.
 *
 * \relates InvocationGroup
 */
DynamicTaskType get_task_type_for_invocation_group(InvocationGroup const &);

/**
 * @brief Reorder a topological ordering of a \ref DynamicOpenDataflowGraph so
 * that the invocations belonging to a pass are adjacent.
 *
 * The result is still a topological ordering: a pass only ever depends on the
 * ones before it, in the order forward, loss, backward, update. What changes is
 * that a topological ordering is otherwise free to interleave them -- an update
 * becomes runnable as soon as the gradient it consumes exists, long before the
 * rest of the backward pass is done -- and interleaved passes fuse into
 * nothing, because a group may not span passes.
 *
 * \relates InvocationGroup
 */
std::vector<DynamicNodeInvocation> sort_invocations_by_pass(
    std::vector<DynamicNodeInvocation> const &topological_order);

/**
 * @brief Split \p execution_order into runs that can each be issued as one
 * Realm task.
 *
 * Fusing removes both the per-task cost of getting a task through the runtime
 * and the gap between one task's kernels finishing and the next task's being
 * submitted: a fused body submits its members' kernels back to back on one
 * stream.
 *
 * An invocation may join a group when it is issued as a plain operator task on
 * exactly one device. That rules out copies, reductions and the parallel
 * operators (the \ref PCGOperatorAttrs that are not also \ref
 * ComputationGraphOpAttrs), which do not become operator tasks at all, and
 * inputs and weights, which have no forward or backward task of their own.
 * Members of a group must additionally agree on their device and on their \ref
 * DynamicTaskType, the latter because the passes can be run separately and
 * select the invocations to issue by task type.
 *
 * Groups are runs of \p execution_order, which is what keeps fusing them from
 * introducing a cycle: any path between two members of a run passes only
 * through nodes that fall between them in a topological ordering, and those
 * are in the run as well.
 *
 * \param max_group_size an upper bound on how many invocations may be fused
 *                       together, or \c std::nullopt for no bound.
 *
 * \relates InvocationGroup
 */
std::vector<InvocationGroup> group_invocations_for_fusion(
    std::vector<PreparedInvocation> const &execution_order,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    std::optional<int> max_group_size);

} // namespace FlexFlow

#endif
