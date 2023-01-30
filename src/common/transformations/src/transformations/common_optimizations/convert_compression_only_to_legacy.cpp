// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"

#include "itt.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

ov::pass::EnableDecompressionConvertConstantFolding::EnableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(EnableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<opset8::Convert>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!ov::is_decompression(node))
            return false;
        enable_constant_folding(node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::pass::ConvertCompressedOnlyToLegacy::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertCompressedOnlyToLegacy);
    if (ov::op::util::has_decompression_converts(f)) {
        Manager manager(get_pass_config());

        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        manager.register_pass<ConvertPrecision>(convert_precision_list);
        using namespace ov::pass;
        REGISTER_PASS(manager, EnableDecompressionConvertConstantFolding)
        REGISTER_PASS(manager, ConstantFolding)

        manager.run_passes(f);
    }
    return false;
}
