// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {
struct OVInferRequestPerfCountersTest : public virtual OVInferRequestTests {
    void SetUp() override;
    ov::InferRequest req;
};
}  // namespace behavior
}  // namespace test
}  // namespace ov
