#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_OPERATION_FUSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_OPERATION_FUSION_H

#include "op-attrs/activation.dtg.h"
#include "op-attrs/pcg_operator_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"

namespace FlexFlow {

/**
 * \brief The \ref Activation \p attrs computes, if it is an elementwise
 * activation that an adjacent operator's kernel could apply itself.
 *
 * \details Returns \c std::nullopt for anything else, including an activation
 * whose parameters cannot be expressed as an \ref Activation (SiLU with a beta
 * other than 1, say), since the fused kernel has nowhere to put them.
 */
std::optional<Activation>
    try_get_fusable_activation(PCGOperatorAttrs const &attrs);

/**
 * \brief The single operator that computes \p producer followed by \p consumer,
 * if some kernel can do that.
 *
 * \details This is the one place that knows which pairs of operators can be
 * fused. Everything else in this file is structural, and asks here whether a
 * pair it has found is a pair anything can actually execute.
 *
 * \note Returning \c std::nullopt is the normal answer for the overwhelming
 * majority of pairs, and means only "not this pair", never "malformed".
 */
std::optional<TrainingOperationAttrs>
    try_fuse_operation_attrs(TrainingOperationAttrs const &producer,
                             TrainingOperationAttrs const &consumer);

/**
 * \brief Whether \p producer and \p consumer can be rewritten as one invocation
 * without changing what the graph computes.
 *
 * \details Independent of whether any kernel exists for the pair; \ref
 * try_fuse_operation_attrs answers that. The conditions are that \p consumer
 * reads \p between and nothing else does, that \p producer produces nothing but
 * \p between (so that folding it away strands no other value), that the two
 * operators run on the same devices, and that their slot names do not collide.
 */
bool invocations_are_structurally_fusible(DynamicNodeInvocation const &producer,
                                          DynamicNodeInvocation const &consumer,
                                          DynamicValueAttrs const &between,
                                          bool between_has_other_readers);

/**
 * \brief The single invocation computing \p producer followed by \p consumer,
 * or \c std::nullopt if the pair cannot be fused.
 *
 * \details The result reads what \p producer read plus whatever \p consumer
 * read besides \p between, and produces what \p consumer produced. \p between
 * appears nowhere in it, which is the point: a value that is not in the graph
 * is not allocated, not written, and not read back.
 */
std::optional<DynamicNodeInvocation>
    try_fuse_invocations(DynamicNodeInvocation const &producer,
                         DynamicNodeInvocation const &consumer,
                         DynamicValueAttrs const &between,
                         bool between_has_other_readers);

/**
 * \brief Fold every adjacent pair of operators that one kernel can compute into
 * a single operator.
 *
 * \details Run before \ref perform_pass_expansion, this is worth more than the
 * forward-pass traffic it saves. Pass expansion gives an operator's backward
 * pass the operator's own inputs and outputs, so a pair fused beforehand gets
 * one backward invocation instead of two, and the value between them is never
 * created in either direction -- the backward kernel recomputes it from what it
 * already reads. Fusing after pass expansion would leave the backward pass
 * exactly as it was.
 *
 * \note Safe to run more than once, and on an already pass-expanded graph:
 * pairs that have been fused no longer look fusible, so a second run finds only
 * what the first could not see (a \ref GradientReductionAttrs and its consumer,
 * say, which does not exist until pass expansion creates it).
 *
 * \param preserved_tensors tensors that must still exist in the result, because
 * something outside the graph will later look them up by \ref
 * dynamic_tensor_guid_t. The loss logit is the important one: \ref
 * perform_loss_insertion runs after this and finds it by guid. Fusing is
 * otherwise free to make an intermediate tensor cease to exist, which is how it
 * saves anything.
 */
DynamicOpenDataflowGraph perform_operation_fusion(
    DynamicOpenDataflowGraph const &,
    std::set<dynamic_tensor_guid_t> const &preserved_tensors);

} // namespace FlexFlow

#endif
