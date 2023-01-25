// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs reduce_sum(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceSum>(node_context);
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
