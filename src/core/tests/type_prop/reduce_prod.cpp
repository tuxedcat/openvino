// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_ops.hpp"

using Type = ::testing::Types<op::v1::ReduceProd>;
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod, ReduceTest, Type);
INSTANTIATE_TYPED_TEST_SUITE_P(type_prop_reduce_prod_et, ReduceArithmeticTest, Type);
