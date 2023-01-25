#include "transformations/common_optimizations/transpose_sinking_utils.hpp"

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "itt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"
#include "transformations/rt_info/transpose_sinking_attr.hpp"

namespace transpose_sinking {

using namespace ov;
using namespace ov::opset9;

using NodePtr = std::shared_ptr<Node>;

Output<Node> ChangeValuesOrder(const Output<Node>& input,
                               const AxisVector& transpose_axis_order,
                               const std::shared_ptr<Constant>& axis) {
    auto rank = transpose_axis_order.size();
    auto split_pad = std::make_shared<Split>(input, axis, rank);
    auto split_outputs = split_pad->outputs();
    OutputVector new_order(split_outputs.size());
    for (size_t i = 0; i < rank; ++i) {
        new_order[i] = split_outputs[transpose_axis_order[i]];
    }
    auto concat_pad = std::make_shared<Concat>(new_order, 0);
    copy_runtime_info(input.get_node_shared_ptr(), {split_pad, concat_pad});
    return concat_pad;
}

TransposeInputsInfo GetFirstTransposeInput(const NodePtr& node) {
    for (size_t input_idx = 0; input_idx < node->get_input_size(); ++input_idx) {
        NodePtr input_node = node->get_input_node_shared_ptr(input_idx);
        auto transpose_node = as_type_ptr<Transpose>(input_node);
        if (!transpose_node)
            continue;
        auto constant_node = as_type_ptr<Constant>(transpose_node->input_value(1).get_node_shared_ptr());
        if (!constant_node)
            continue;
        {
            TransposeInputsInfo input_info;
            input_info.transpose = transpose_node;
            input_info.transpose_const = constant_node;
            input_info.input_idx = input_idx;
            return input_info;
        }
    }

    return {};
}

bool IfNodeHasTransposeInputs(const Output<Node>& output) {
    TransposeInputsInfo inputs_info = GetFirstTransposeInput(output.get_node_shared_ptr());
    return !inputs_info.isEmpty();
}

AxisVector ReverseTransposeOrder(const AxisVector& axis_order) {
    AxisVector out(axis_order.size());
    for (size_t i = 0; i < axis_order.size(); i++) {
        out.at(axis_order[i]) = i;
    }
    return out;
}

void SwapOutputNames(Output<Node> output1, Output<Node> output2) {
    const auto node2_output_names = output2.get_names();
    output2.set_names(output1.get_names());
    output1.set_names(node2_output_names);
}

void SwapFriendlyNames(const NodePtr& node1, const NodePtr& node2) {
    const std::string node2_name = node2->get_friendly_name();
    node2->set_friendly_name(node1->get_friendly_name());
    node1->set_friendly_name(node2_name);
}

void SwapNames(const NodePtr& node1, const NodePtr& node2) {
    SwapFriendlyNames(node1, node2);
    SwapOutputNames(node1->output(0), node2->output(0));
}

namespace {

bool HasDynamicRankInput(const NodePtr& node) {
    for (auto& input_node : node->input_values()) {
        const ov::Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return true;
    }
    return false;
}

ov::Rank::value_type GetMaxInputRank(const NodePtr& node) {
    ov::Rank::value_type max_input_rank = 0;
    for (auto& input_node : node->input_values()) {
        const ov::Rank output_rank = input_node.get_partial_shape().rank();
        if (output_rank.is_dynamic())
            return -1;
        const ov::Rank::value_type output_rank_len = output_rank.get_length();
        if (output_rank_len > max_input_rank)
            max_input_rank = output_rank_len;
    }
    return max_input_rank;
}

NodePtr InsertUnsqueeze(const Output<Node>& node, size_t n_dims) {
    std::vector<size_t> dims(n_dims);
    std::iota(dims.begin(), dims.end(), 0);
    auto unsqueeze_const = std::make_shared<Constant>(ov::element::i64, Shape{dims.size()}, dims);
    auto unsqueeze = std::make_shared<Unsqueeze>(node, unsqueeze_const);
    copy_runtime_info(node.get_node_shared_ptr(), {unsqueeze, unsqueeze_const});
    return unsqueeze;
}

ov::Output<ov::Node> FixInputNodeRank(ov::Output<ov::Node> input_node, ov::Rank::value_type required_rank) {
    const ov::Rank::value_type output_rank = input_node.get_partial_shape().rank().get_length();
    if (output_rank >= required_rank)
        return input_node;
    return InsertUnsqueeze(input_node, required_rank - output_rank)->output(0);
}

}  // namespace

namespace sink_forward {

void UpdateInputTransposes(const NodePtr& main_node, const TransposeInputsInfo& transpose_input_info) {
    if (transpose_input_info.isEmpty() || HasDynamicRankInput(main_node))
        return;

    const auto max_input_rank = GetMaxInputRank(main_node);
    if (max_input_rank < 0)
        return;

    const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
    const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
    const size_t tranpose_input_index = transpose_input_info.input_idx;
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    for (size_t i = 0; i < main_node->get_input_size(); ++i) {
        auto input_node = main_node->input_value(i);
        if (i == tranpose_input_index) {
            auto transpose_parent = input_node.get_node()->input_value(0);
            main_node->input(i).replace_source_output(transpose_parent);
        } else {
            input_node = FixInputNodeRank(input_node, max_input_rank);

            auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                                  Shape{reversed_traspose_axis_order.size()},
                                                                  reversed_traspose_axis_order);
            auto new_transpose = std::make_shared<Transpose>(input_node, new_transpose_const);

            main_node->input(i).replace_source_output(new_transpose->output(0));

            copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});
        }
    }
}

void RemoveInputNode(const NodePtr& main_node, size_t input_idx) {
    auto input_node = main_node->input_value(input_idx);
    if (input_node.get_node()->get_input_size() < (input_idx + 1))
        return;
    auto parent_node = input_node.get_node()->input_value(input_idx);
    main_node->input(input_idx).replace_source_output(parent_node);
}

NodeVector InsertOutputTransposes(const NodePtr& main_node, const TransposeInputsInfo& transpose_input_info) {
    if (transpose_input_info.isEmpty())
        return {};
    const auto transpose_axis_order = transpose_input_info.transpose_const->get_axis_vector_val();
    const auto reversed_traspose_axis_order = ReverseTransposeOrder(transpose_axis_order);
    const auto transpose_element_type = transpose_input_info.transpose_const->get_element_type();

    NodeVector new_nodes;

    for (size_t i = 0; i < main_node->get_output_size(); ++i) {
        auto main_node_consumers = main_node->output(i).get_target_inputs();

        auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                              Shape{transpose_axis_order.size()},
                                                              transpose_axis_order);
        auto new_transpose = std::make_shared<Transpose>(main_node->output(i), new_transpose_const);

        for (auto& consumer : main_node_consumers) {
            consumer.replace_source_output(new_transpose);
        }

        copy_runtime_info(main_node, {new_transpose, new_transpose_const});
        SwapOutputNames(main_node->output(i), new_transpose->output(0));

        if (main_node->get_output_size() > 1)
            new_transpose->set_friendly_name(main_node->get_friendly_name() + "." + std::to_string(i));
        else
            SwapFriendlyNames(new_transpose, main_node);

        new_nodes.push_back(new_transpose);
    }

    return new_nodes;
}

}  // namespace sink_forward

namespace sink_backward {

NodeVector InsertTransposeBeforeNode(const NodePtr& main_node,
                                     const std::shared_ptr<Constant>& transpose_const,
                                     std::vector<int> input_indexes) {
    if (input_indexes.empty()) {
        input_indexes.resize(main_node->get_input_size());
        std::iota(input_indexes.begin(), input_indexes.end(), 0);
    }
    const auto transpose_axis_order = transpose_const->get_axis_vector_val();
    const auto transpose_element_type = transpose_const->get_element_type();

    if (HasDynamicRankInput(main_node))
        return {};

    NodeVector new_nodes;

    const auto max_input_rank = GetMaxInputRank(main_node);
    if (max_input_rank < 0)
        return {};

    for (const auto& i : input_indexes) {
        auto input_node = FixInputNodeRank(main_node->input_value(i), max_input_rank);

        auto new_transpose_const = std::make_shared<Constant>(transpose_element_type,
                                                              Shape{transpose_axis_order.size()},
                                                              transpose_axis_order);
        auto new_transpose = std::make_shared<Transpose>(input_node, new_transpose_const);

        main_node->input(i).replace_source_output(new_transpose->output(0));

        copy_runtime_info(input_node.get_node_shared_ptr(), {new_transpose, new_transpose_const});

        new_nodes.push_back(new_transpose);
    }

    return new_nodes;
}
}  // namespace sink_backward

#define CHECK_TRANSPOSE_SINKING_SUPPORTED(TYPE, node) \
    if (dynamic_cast<TYPE*>(node)) {                  \
        return true;                                  \
    }

namespace {

bool CanPropagateForwardThrough(Node* node) {
    CHECK_TRANSPOSE_SINKING_SUPPORTED(ov::op::util::UnaryElementwiseArithmetic, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Clamp, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Elu, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(SoftPlus, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(LogicalNot, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Convert, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(ov::op::util::BinaryElementwiseArithmetic, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Concat, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Split, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(Transpose, node);
    CHECK_TRANSPOSE_SINKING_SUPPORTED(PRelu, node);

    return false;
}

bool CanPropagateForward(const NodePtr& node) {
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        for (auto& consumer_input : node->output(i).get_target_inputs()) {
            if (!CanPropagateForwardThrough(consumer_input.get_node()))
                return false;
        }
    }

    return true;
}

}  // namespace

void UpdateForwardSinkingAbility(const NodePtr& node) {
    if (!CanPropagateForward(node))
        mark_as_no_sinking_node(node);
}

namespace {

std::shared_ptr<Constant> GetTransposeConstant(Node* node) {
    auto transpose_node = dynamic_cast<Transpose*>(node);
    if (!transpose_node)
        return {};

    auto constant_node = as_type_ptr<Constant>(transpose_node->input_value(1).get_node_shared_ptr());
    if (!constant_node)
        return {};

    return constant_node;
}

Node* FindFirstConsumer(const NodePtr& node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        auto inputs = node->get_output_target_inputs(output_idx);
        if (inputs.empty())
            continue;
        return inputs.begin()->get_node();
    }
    return nullptr;
}

bool HasSameOutputTransposeNodes(const NodePtr& main_node) {
    AxisVector first_transpose_axis_order;
    {
        Node* first_consumer = FindFirstConsumer(main_node);
        if (!first_consumer)
            return false;
        auto constant_node = GetTransposeConstant(first_consumer);
        if (!constant_node)
            return false;
        first_transpose_axis_order = constant_node->get_axis_vector_val();
    }

    for (size_t output_idx = 0; output_idx < main_node->get_output_size(); ++output_idx) {
        for (auto& input : main_node->get_output_target_inputs(output_idx)) {
            auto constant_node = GetTransposeConstant(input.get_node());
            if (!constant_node)
                return false;

            AxisVector transpose_axis_order = constant_node->get_axis_vector_val();
            if (transpose_axis_order.size() != first_transpose_axis_order.size())
                return false;
            if (!std::equal(transpose_axis_order.begin(),
                            transpose_axis_order.end(),
                            first_transpose_axis_order.begin()))
                return false;
        }
    }

    return true;
}

}  // namespace

bool HasSameOutputTransposeNodes(const Output<Node>& output) {
    return HasSameOutputTransposeNodes(output.get_node_shared_ptr());
}

void RemoveSingleOutputConsumers(const NodePtr& node) {
    for (size_t output_idx = 0; output_idx < node->get_output_size(); ++output_idx) {
        for (auto& input : node->get_output_target_inputs(output_idx)) {
            Node* consumer = input.get_node();
            if (consumer->get_output_size() != 1)
                continue;
            consumer->output(0).replace(node->output(output_idx));
        }
    }
}

}  // namespace transpose_sinking
