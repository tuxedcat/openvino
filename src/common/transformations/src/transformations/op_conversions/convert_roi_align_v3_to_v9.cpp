// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_roi_align_v3_to_v9.hpp"

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset9.hpp>

#include "itt.hpp"

ov::pass::ConvertROIAlign3To9::ConvertROIAlign3To9() {
    MATCHER_SCOPE(ConvertROIAlign3To9);

    auto roi_align_v3 = pattern::wrap_type<ov::opset3::ROIAlign>();

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto roi_align_v3_node = std::dynamic_pointer_cast<ov::opset3::ROIAlign>(m.get_match_root());
        if (!roi_align_v3_node)
            return false;

        const int pooled_h = roi_align_v3_node->get_pooled_h();
        const int pooled_w = roi_align_v3_node->get_pooled_w();
        const int sampling_ratio = roi_align_v3_node->get_sampling_ratio();
        const float spatial_scale = roi_align_v3_node->get_spatial_scale();
        ov::opset3::ROIAlign::PoolingMode m_mode_v3 = roi_align_v3_node->get_mode();
        ov::opset9::ROIAlign::PoolingMode m_mode_v9;
        switch (m_mode_v3) {
        case ov::opset3::ROIAlign::PoolingMode::AVG: {
            m_mode_v9 = ov::opset9::ROIAlign::PoolingMode::AVG;
            break;
        }
        case ov::opset3::ROIAlign::PoolingMode::MAX: {
            m_mode_v9 = ov::opset9::ROIAlign::PoolingMode::MAX;
            break;
        }
        default: {
            throw Exception("unsupported PoolingMode ");
        }
        }

        auto roi_align_v9 = std::make_shared<ov::opset9::ROIAlign>(roi_align_v3_node->input_value(0),
                                                                   roi_align_v3_node->input_value(1),
                                                                   roi_align_v3_node->input_value(2),
                                                                   pooled_h,
                                                                   pooled_w,
                                                                   sampling_ratio,
                                                                   spatial_scale,
                                                                   m_mode_v9);
        roi_align_v9->set_friendly_name(roi_align_v3_node->get_friendly_name());
        ngraph::copy_runtime_info(roi_align_v3_node, roi_align_v9);
        ngraph::replace_node(roi_align_v3_node, roi_align_v9);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(roi_align_v3, matcher_name);
    register_matcher(m, callback);
}
