#include "task-spec/dynamic_graph/operation_fusion.h"
#include "op-attrs/ops/element_unary.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

namespace {

DynamicTensorSlot mk_slot(TensorSlotName slot_name) {
  return DynamicTensorSlot{
      /*slot_name=*/slot_name,
      /*slot_tensor_role=*/std::nullopt,
      /*task_shard=*/std::nullopt,
  };
}

DynamicNodeAttrs mk_node_attrs(size_t layer_guid,
                               PCGOperatorAttrs const &op_attrs) {
  return DynamicNodeAttrs{
      /*task_type=*/std::nullopt,
      /*device_ids=*/std::nullopt,
      /*mapping=*/std::nullopt,
      /*op_attrs=*/TrainingOperationAttrs{op_attrs},
      /*layer_guid=*/
      dynamic_layer_guid_t{parallel_layer_guid_t{Node{layer_guid}}},
      /*per_device_op_state=*/std::nullopt,
  };
}

DynamicValueAttrs mk_value_attrs(size_t src_layer_guid,
                                 TensorSlotName src_slot) {
  return DynamicValueAttrs{
      /*tensor_guid=*/dynamic_tensor_guid_t{parallel_tensor_guid_t{
          KwargDataflowOutput{Node{src_layer_guid}, src_slot}}},
      /*parallel_tensor_shape=*/std::nullopt,
      /*create_grad=*/true,
      /*subgradient_id=*/std::nullopt,
      /*shard_coord=*/std::nullopt,
      /*mapping=*/std::nullopt,
      /*accessor=*/std::nullopt,
      /*role=*/std::nullopt,
  };
}

TensorShape mk_input_shape() {
  return TensorShape{
      TensorDims{FFOrdered<positive_int>{2_p, 4_p, 8_p, 8_p}},
      DataType::FLOAT,
  };
}

BatchNormAttrs
    mk_batch_norm_attrs(std::optional<Activation> activation = std::nullopt,
                        BatchNormMode mode = BatchNormMode::SPATIAL) {
  return BatchNormAttrs{
      /*activation=*/activation,
      /*affine=*/true,
      /*eps=*/1e-5,
      /*momentum=*/std::nullopt,
      /*mode=*/mode,
  };
}

ElementUnaryAttrs mk_silu_attrs(std::optional<float> scalar = std::nullopt) {
  return ElementUnaryAttrs{
      /*op_type=*/OperatorType::SILU,
      /*scalar=*/scalar,
  };
}

/**
 * @brief input -> batch_norm -> consumer, the shape every fusion candidate in
 * YOLOv10x has.
 */
struct Chain {
  DynamicOpenDataflowGraph g;
  DynamicValueAttrs input_output;
  DynamicValueAttrs batch_norm_output;
  DynamicValueAttrs consumer_output;
};

Chain mk_chain(PCGOperatorAttrs const &batch_norm_op,
               PCGOperatorAttrs const &consumer_op,
               std::set<DynamicNodeInvocation> const &extra_invocations = {}) {
  DynamicValueAttrs input_output = mk_value_attrs(1, TensorSlotName::OUTPUT);
  DynamicValueAttrs gamma = mk_value_attrs(2, TensorSlotName::OUTPUT);
  DynamicValueAttrs beta = mk_value_attrs(3, TensorSlotName::OUTPUT);
  DynamicValueAttrs batch_norm_output =
      mk_value_attrs(4, TensorSlotName::OUTPUT);
  DynamicValueAttrs consumer_output = mk_value_attrs(5, TensorSlotName::OUTPUT);

  auto mk_source = [](size_t guid, DynamicValueAttrs const &out) {
    return DynamicNodeInvocation{
        /*inputs=*/{},
        /*node_attrs=*/
        mk_node_attrs(guid, PCGOperatorAttrs{InputAttrs{mk_input_shape()}}),
        /*outputs=*/{{mk_slot(TensorSlotName::OUTPUT), out}},
    };
  };

  DynamicNodeInvocation batch_norm_invocation = DynamicNodeInvocation{
      /*inputs=*/{{mk_slot(TensorSlotName::INPUT), input_output},
                  {mk_slot(TensorSlotName::GAMMA), gamma},
                  {mk_slot(TensorSlotName::BETA), beta}},
      /*node_attrs=*/mk_node_attrs(4, batch_norm_op),
      /*outputs=*/{{mk_slot(TensorSlotName::OUTPUT), batch_norm_output}},
  };

  DynamicNodeInvocation consumer_invocation = DynamicNodeInvocation{
      /*inputs=*/{{mk_slot(TensorSlotName::INPUT), batch_norm_output}},
      /*node_attrs=*/mk_node_attrs(5, consumer_op),
      /*outputs=*/{{mk_slot(TensorSlotName::OUTPUT), consumer_output}},
  };

  std::set<DynamicNodeInvocation> invocations = {
      mk_source(1, input_output),
      mk_source(2, gamma),
      mk_source(3, beta),
      batch_norm_invocation,
      consumer_invocation,
  };
  invocations.insert(extra_invocations.begin(), extra_invocations.end());

  return Chain{
      dynamic_open_dataflow_graph_from_invocation_set(invocations),
      input_output,
      batch_norm_output,
      consumer_output,
  };
}

std::set<TrainingOperationAttrs>
    op_attrs_of(DynamicOpenDataflowGraph const &g) {
  std::set<TrainingOperationAttrs> result;
  for (DynamicNodeInvocation const &i : get_dynamic_invocation_set(g)) {
    result.insert(i.node_attrs.op_attrs.value());
  }
  return result;
}

} // namespace

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("try_get_fusable_activation") {
    SUBCASE("an activation with no parameters is fusable") {
      CHECK(try_get_fusable_activation(PCGOperatorAttrs{mk_silu_attrs()}) ==
            std::optional<Activation>{Activation::SILU});
      CHECK(try_get_fusable_activation(PCGOperatorAttrs{make_relu_attrs()}) ==
            std::optional<Activation>{Activation::RELU});
    }

    SUBCASE("SiLU's beta is fusable only at its default of 1") {
      CHECK(try_get_fusable_activation(PCGOperatorAttrs{mk_silu_attrs(1.0f)}) ==
            std::optional<Activation>{Activation::SILU});
      CHECK(try_get_fusable_activation(PCGOperatorAttrs{mk_silu_attrs(2.0f)}) ==
            std::optional<Activation>{});
    }

    SUBCASE("an elementwise operator that is not an activation") {
      ElementUnaryAttrs exp_attrs = ElementUnaryAttrs{
          /*op_type=*/OperatorType::EXP,
          /*scalar=*/std::nullopt,
      };
      CHECK(try_get_fusable_activation(PCGOperatorAttrs{exp_attrs}) ==
            std::optional<Activation>{});
    }

    SUBCASE("an operator that is not elementwise") {
      CHECK(try_get_fusable_activation(PCGOperatorAttrs{
                mk_batch_norm_attrs()}) == std::optional<Activation>{});
    }
  }

  TEST_CASE("try_fuse_operation_attrs") {
    auto training = [](PCGOperatorAttrs const &a) {
      return TrainingOperationAttrs{a};
    };

    SUBCASE("batch norm followed by an activation") {
      std::optional<TrainingOperationAttrs> result = try_fuse_operation_attrs(
          training(PCGOperatorAttrs{mk_batch_norm_attrs()}),
          training(PCGOperatorAttrs{mk_silu_attrs()}));

      CHECK(result ==
            std::optional<TrainingOperationAttrs>{training(
                PCGOperatorAttrs{mk_batch_norm_attrs(Activation::SILU)})});
    }

    SUBCASE("a batch norm that already has an activation is left alone") {
      CHECK(
          try_fuse_operation_attrs(
              training(PCGOperatorAttrs{mk_batch_norm_attrs(Activation::RELU)}),
              training(PCGOperatorAttrs{mk_silu_attrs()})) ==
          std::optional<TrainingOperationAttrs>{});
    }

    SUBCASE("an activation no batch-norm kernel can apply") {
      ElementUnaryAttrs gelu_attrs = ElementUnaryAttrs{
          /*op_type=*/OperatorType::GELU,
          /*scalar=*/std::nullopt,
      };

      // Fusable in the sense that it is a parameterless activation, so the
      // refusal has to come from what the batch-norm kernels implement.
      REQUIRE(try_get_fusable_activation(PCGOperatorAttrs{gelu_attrs}) ==
              std::optional<Activation>{Activation::GELU});

      CHECK(try_fuse_operation_attrs(
                training(PCGOperatorAttrs{mk_batch_norm_attrs()}),
                training(PCGOperatorAttrs{gelu_attrs})) ==
            std::optional<TrainingOperationAttrs>{});
    }

    SUBCASE("a mode whose kernel cannot apply an activation") {
      CHECK(try_fuse_operation_attrs(
                training(PCGOperatorAttrs{mk_batch_norm_attrs(
                    std::nullopt, BatchNormMode::PER_ACTIVATION)}),
                training(PCGOperatorAttrs{mk_silu_attrs()})) ==
            std::optional<TrainingOperationAttrs>{});
    }

    SUBCASE("a producer no kernel can fuse into") {
      CHECK(try_fuse_operation_attrs(
                training(PCGOperatorAttrs{ConcatAttrs{
                    /*axis=*/ff_dim_t{0_n},
                    /*num_inputs=*/int_ge_two{2},
                }}),
                training(PCGOperatorAttrs{mk_silu_attrs()})) ==
            std::optional<TrainingOperationAttrs>{});
    }

    SUBCASE("a consumer that is not an activation") {
      CHECK(try_fuse_operation_attrs(
                training(PCGOperatorAttrs{mk_batch_norm_attrs()}),
                training(PCGOperatorAttrs{mk_batch_norm_attrs()})) ==
            std::optional<TrainingOperationAttrs>{});
    }

    SUBCASE("the order matters") {
      CHECK(try_fuse_operation_attrs(
                training(PCGOperatorAttrs{mk_silu_attrs()}),
                training(PCGOperatorAttrs{mk_batch_norm_attrs()})) ==
            std::optional<TrainingOperationAttrs>{});
    }
  }

  TEST_CASE("perform_operation_fusion") {
    SUBCASE("batch norm and the activation reading it become one operator") {
      Chain chain = mk_chain(PCGOperatorAttrs{mk_batch_norm_attrs()},
                             PCGOperatorAttrs{mk_silu_attrs()});

      DynamicOpenDataflowGraph result = perform_operation_fusion(chain.g, {});

      SUBCASE("one fewer invocation") {
        CHECK(dynamic_graph_num_nodes(result) + 1_n ==
              dynamic_graph_num_nodes(chain.g));
      }

      SUBCASE("the batch norm carries the activation") {
        CHECK(contains(op_attrs_of(result),
                       TrainingOperationAttrs{PCGOperatorAttrs{
                           mk_batch_norm_attrs(Activation::SILU)}}));
      }

      SUBCASE("the activation operator is gone") {
        CHECK(!contains(
            op_attrs_of(result),
            TrainingOperationAttrs{PCGOperatorAttrs{mk_silu_attrs()}}));
      }

      SUBCASE("the value between them is gone") {
        CHECK(!contains(dynamic_graph_get_internal_values(result),
                        chain.batch_norm_output));
      }

      SUBCASE("the fused operator produces what the activation produced") {
        CHECK(contains(dynamic_graph_get_internal_values(result),
                       chain.consumer_output));
      }

      SUBCASE("the fused operator reads what the batch norm read") {
        CHECK(contains(dynamic_graph_get_internal_values(result),
                       chain.input_output));
      }
    }

    SUBCASE("a preserved tensor is not fused away") {
      Chain chain = mk_chain(PCGOperatorAttrs{mk_batch_norm_attrs()},
                             PCGOperatorAttrs{mk_silu_attrs()});

      DynamicOpenDataflowGraph result = perform_operation_fusion(
          chain.g, {chain.batch_norm_output.tensor_guid});

      CHECK(result == chain.g);
    }

    SUBCASE("a value with a second reader is not fused away") {
      DynamicValueAttrs input_output =
          mk_value_attrs(1, TensorSlotName::OUTPUT);
      DynamicValueAttrs batch_norm_output =
          mk_value_attrs(4, TensorSlotName::OUTPUT);
      DynamicValueAttrs second_reader_output =
          mk_value_attrs(6, TensorSlotName::OUTPUT);

      DynamicNodeInvocation second_reader = DynamicNodeInvocation{
          /*inputs=*/{{mk_slot(TensorSlotName::INPUT), batch_norm_output}},
          /*node_attrs=*/
          mk_node_attrs(6, PCGOperatorAttrs{make_relu_attrs()}),
          /*outputs=*/
          {{mk_slot(TensorSlotName::OUTPUT), second_reader_output}},
      };

      Chain chain = mk_chain(PCGOperatorAttrs{mk_batch_norm_attrs()},
                             PCGOperatorAttrs{mk_silu_attrs()},
                             {second_reader});

      CHECK(perform_operation_fusion(chain.g, {}) == chain.g);
    }

    SUBCASE("a pair no kernel can fuse is left alone") {
      Chain chain = mk_chain(PCGOperatorAttrs{mk_batch_norm_attrs()},
                             PCGOperatorAttrs{mk_batch_norm_attrs()});

      CHECK(perform_operation_fusion(chain.g, {}) == chain.g);
    }

    SUBCASE("an operator with no consumer is left alone") {
      DynamicValueAttrs input_output =
          mk_value_attrs(1, TensorSlotName::OUTPUT);
      DynamicValueAttrs batch_norm_output =
          mk_value_attrs(4, TensorSlotName::OUTPUT);

      DynamicOpenDataflowGraph g =
          dynamic_open_dataflow_graph_from_invocation_set({
              DynamicNodeInvocation{
                  /*inputs=*/{},
                  /*node_attrs=*/
                  mk_node_attrs(1,
                                PCGOperatorAttrs{InputAttrs{mk_input_shape()}}),
                  /*outputs=*/
                  {{mk_slot(TensorSlotName::OUTPUT), input_output}},
              },
              DynamicNodeInvocation{
                  /*inputs=*/{{mk_slot(TensorSlotName::INPUT), input_output}},
                  /*node_attrs=*/
                  mk_node_attrs(4, PCGOperatorAttrs{mk_batch_norm_attrs()}),
                  /*outputs=*/
                  {{mk_slot(TensorSlotName::OUTPUT), batch_norm_output}},
              },
          });

      CHECK(perform_operation_fusion(g, {}) == g);
    }

    SUBCASE("running it again changes nothing") {
      Chain chain = mk_chain(PCGOperatorAttrs{mk_batch_norm_attrs()},
                             PCGOperatorAttrs{mk_silu_attrs()});

      DynamicOpenDataflowGraph once = perform_operation_fusion(chain.g, {});
      DynamicOpenDataflowGraph twice = perform_operation_fusion(once, {});

      CHECK(twice == once);
    }

    SUBCASE("a graph with nothing to fuse is returned unchanged") {
      Chain chain =
          mk_chain(PCGOperatorAttrs{mk_batch_norm_attrs(Activation::SILU)},
                   PCGOperatorAttrs{mk_silu_attrs()});

      CHECK(perform_operation_fusion(chain.g, {}) == chain.g);
    }
  }
}
