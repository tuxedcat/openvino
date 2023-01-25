// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/convolution.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Convolution;
using ov::op::v1::ConvolutionBackpropData;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
