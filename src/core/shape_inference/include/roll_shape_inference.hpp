// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/roll.hpp>

#include "utils.hpp"

namespace ov {
namespace op {
namespace v7 {

template <class TShape>
std::vector<TShape> shape_infer(const Roll* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);

    const auto& data_pshape = input_shapes[0];
    const auto& shift_pshape = input_shapes[1];
    const auto& axes_pshape = input_shapes[2];

    if (shift_pshape.rank().is_static()) {
        const auto& shift_rank = shift_pshape.size();
        NODE_VALIDATION_CHECK(op, shift_rank <= 1, "Shift must be a scalar or 1D tensor.");
        // If shift is a scalar, than axes can be arbitrary 1d tensor and we don't need
        // to check shift shape consistency with axes, otherwise the check is needed.
        if (shift_rank == 1) {
            NODE_VALIDATION_CHECK(op,
                                  shift_pshape.compatible(axes_pshape),
                                  "If shift is a 1D vector, axes must be a 1D tensor of the same size.");
        }
    }

    NODE_VALIDATION_CHECK(op,
                          axes_pshape.rank().is_dynamic() || axes_pshape.size() <= 1,
                          "Axes must be a scalar or 1D tensor.");

    if (data_pshape.rank().is_static()) {
        if (const auto& axes = get_input_const_data_as<TShape, int64_t>(op, 2, constant_data)) {
            ov::normalize_axes(op, data_pshape.size(), *axes);
        }
    }

    return {data_pshape};
}

template <class TShape>
void shape_infer(const Roll* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = shape_infer<TShape>(op, input_shapes, constant_data);
}

}  // namespace v7
}  // namespace op
}  // namespace ov
