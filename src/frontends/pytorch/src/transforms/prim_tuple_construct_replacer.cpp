// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prim_tuple_construct_replacer.hpp"

#include <openvino/frontend/pytorch/decoder.hpp>
#include <openvino/op/util/framework_node.hpp>
#include <openvino/opsets/opset10.hpp>

#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

bool DecomposeTupleResults::run_on_model(const std::shared_ptr<Model>& model) {
    bool at_least_one_decomposed = false;

    ResultVector results = model->get_results();

    for (size_t i = 0; i < results.size(); ++i) {
        auto result = results[i];
        auto input_node = result->get_input_node_shared_ptr(0);
        auto tuple_construct = cast_fw_node(input_node, "prim::TupleConstruct");
        if (!tuple_construct) {
            continue;
        }
        auto inputs = input_node->inputs();
        for (auto input : inputs) {
            model->add_results({std::make_shared<opset10::Result>(input.get_source_output())});
        }

        model->remove_result(result);
        at_least_one_decomposed = true;
    }

    return at_least_one_decomposed;
};
}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
