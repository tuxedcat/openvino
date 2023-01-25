// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/pattern/matcher.hpp>

namespace ngraph {
namespace snippets {
namespace pass {

/**
 * @interface InsertLoad
 * @brief Inserts explicit load instruction after each parameter and buffer.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertLoad: public ngraph::pass::MatcherPass {
public:
    InsertLoad(const size_t count = 1lu);
};

/**
 * @interface InsertStore
 * @brief Inserts explicit store instruction before each result and buffer.
 * The pass is used to convert model to a canonical form for code generation
 * @ingroup snippets
 */
class InsertStore: public ngraph::pass::MatcherPass {
public:
    InsertStore(const size_t count = 1lu);
};


}  // namespace pass
}  // namespace snippets
}  // namespace ngraph
