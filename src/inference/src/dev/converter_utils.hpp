// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace legacy_convert {

void fill_input_info(const ov::Output<const ov::Node>& input, InferenceEngine::InputInfo::Ptr& inputInfo);
void fill_output_info(const ov::Output<const ov::Node>& output, InferenceEngine::DataPtr& outputInfo);

InferenceEngine::CNNNetwork convert_model(const std::shared_ptr<const ov::Model>& model, bool is_new_api);
std::shared_ptr<const ov::Model> convert_model(const InferenceEngine::CNNNetwork& model, bool is_new_api);

std::shared_ptr<::InferenceEngine::IInferencePlugin> convert_plugin(const std::shared_ptr<::ov::IPlugin>& plugin);
std::shared_ptr<::ov::IPlugin> convert_plugin(const std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin);

}  // namespace legacy_convert
}  // namespace ov

