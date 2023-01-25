// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "cnn_network_ngraph_impl.hpp"

namespace ov {
namespace details {
/**
 * @brief Checks if the input network is batch-able (e.g. no dynamic inputs, inputs has the batch dimension, etc)
 * @param function A ngraph function to check for automatic-batching applicability
 * @return An enum value indicating whether the network can be safely batched (with HETERO or as is) or not
 */
enum class NetworkBatchAbility : uint32_t { NO = 0, AS_IS, WITH_HETERO };
NetworkBatchAbility is_model_batchable(const std::shared_ptr<const ov::Model>& model,
                                       const std::string& deviceNoBatch,
                                       bool strictly_track_dims);

}  // namespace details
}  // namespace ov
