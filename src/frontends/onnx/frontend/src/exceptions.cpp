// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "exceptions.hpp"

#include <sstream>

namespace ngraph {
namespace onnx_import {
namespace error {
namespace detail {
std::string get_error_msg_prefix(const Node& node) {
    std::stringstream ss;
    ss << "While validating ONNX node '" << node << "'";
    return ss.str();
}
}  // namespace detail
}  // namespace error
}  // namespace onnx_import
}  // namespace ngraph
