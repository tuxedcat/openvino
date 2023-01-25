// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/pattern/op/or.hpp>
#include <transformations/utils/utils.hpp>
#include <utility>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset9.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/log.hpp"

namespace transpose_sinking {

struct TransposeInputsInfo {
    std::shared_ptr<ov::opset9::Transpose> transpose;
    std::shared_ptr<ov::opset9::Constant> transpose_const;
    size_t input_idx;

    bool isEmpty() const {
        return !transpose || !transpose_const;
    }
};

/**
 * @brief Finds node first input that is a transpose operation and returns filled TransposeInputsInfo
 * for it
 */
TransposeInputsInfo GetFirstTransposeInput(const std::shared_ptr<ov::Node>&);

/**
 * @brief Checks if @arg has any input node that is a transpose operation
 */
bool IfNodeHasTransposeInputs(const ov::Output<ov::Node>&);

/**
 * @brief Reverses order of transpose operation. Do it in a such way that if we had couple following one after
 * another transposes (one would be reversed version of another) we will have no transpose as a result of that
 * couple of transposes.
 */
ov::AxisVector ReverseTransposeOrder(const ov::AxisVector&);

/**
 * @brief Swaps @args output tensor names
 */
void SwapOutputNames(ov::Output<ov::Node>, ov::Output<ov::Node>);

/**
 * @brief Swaps @args friendly names
 */
void SwapFriendlyNames(const std::shared_ptr<ov::Node>&, const std::shared_ptr<ov::Node>&);

/**
 * @brief Swaps @args output tensor names and friendly names
 */
void SwapNames(const std::shared_ptr<ov::Node>&, const std::shared_ptr<ov::Node>&);

namespace sink_forward {
/**
 * @brief Inserts reversed transposed on @args main_node inputs. Removes input transpose specified in @arg
 * transpose_input_info
 */
void UpdateInputTransposes(const std::shared_ptr<ov::Node>& main_node, const TransposeInputsInfo& transpose_input_info);

/**
 * @brief Removes @arg input node
 */
void RemoveInputNode(const std::shared_ptr<ov::Node>&, size_t input_idx);

/**
 * @brief Inserts transposes on each main_node output with the order specified in @arg transpose_input_info
 */
ov::NodeVector InsertOutputTransposes(const std::shared_ptr<ov::Node>& main_node,
                                      const TransposeInputsInfo& transpose_input_info);
}  // namespace sink_forward

namespace sink_backward {
/**
 * @brief Inserts transposes on inputs of @arg main_node specified by @arg input_indexes
 * with the order specified in @arg transpose_const. If @arg input_indexes is empty, then it inserts
 * transposes for all inputs.
 */
ov::NodeVector InsertTransposeBeforeNode(const std::shared_ptr<ov::Node>& main_node,
                                         const std::shared_ptr<ov::opset9::Constant>& transpose_const,
                                         std::vector<int> input_indexes = {});
}  // namespace sink_backward

void UpdateForwardSinkingAbility(const std::shared_ptr<ov::Node>&);

/**
 *  @brief Checks if @arg has consumers that all are the same transpose operation. If no consumers at all
 *  returns false.
 */
bool HasSameOutputTransposeNodes(const ov::Output<ov::Node>&);

/**
 * Removes all direct node consumers that have one output
 */
void RemoveSingleOutputConsumers(const std::shared_ptr<ov::Node>&);

/**
 * Changes the order of values in @arg input according to @arg transpose_axis_order along @arg axis
 */
ov::Output<ov::Node> ChangeValuesOrder(const ov::Output<ov::Node>& input,
                                       const ov::AxisVector& transpose_axis_order,
                                       const std::shared_ptr<ov::opset9::Constant>& axis);
}  // namespace transpose_sinking
