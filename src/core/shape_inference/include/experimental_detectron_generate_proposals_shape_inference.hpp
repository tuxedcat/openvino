// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/experimental_detectron_generate_proposals.hpp>

namespace ov {
namespace op {
namespace v6 {

template <class T>
void shape_infer(const ExperimentalDetectronGenerateProposalsSingleImage* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 4 && output_shapes.size() == 2);
    auto post_nms_count = static_cast<size_t>(op->get_attrs().post_nms_count);

    const auto& im_info_shape = input_shapes[0];
    const auto& anchors_shape = input_shapes[1];
    const auto& deltas_shape = input_shapes[2];
    const auto& scores_shape = input_shapes[3];
    const auto im_info_shape_rank = im_info_shape.rank();
    NODE_VALIDATION_CHECK(op,
                          im_info_shape_rank.compatible(1),
                          "The 'input_im_info' input is expected to be a 1D. Got: ",
                          im_info_shape);

    if (im_info_shape_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              im_info_shape[0].compatible(3),
                              "The 'input_im_info' shape is expected to be a compatible with [3]. Got: ",
                              im_info_shape);
    }

    const auto anchors_shape_rank = anchors_shape.rank();
    NODE_VALIDATION_CHECK(op,
                          anchors_shape_rank.compatible(2),
                          "The 'input_anchors' input is expected to be a 2D. Got: ",
                          anchors_shape);
    if (anchors_shape_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              anchors_shape[1].compatible(4),
                              "The second dimension of 'input_anchors' should be compatible with 4. Got: ",
                              anchors_shape[1]);
    }
    const auto deltas_shape_rank = deltas_shape.rank();
    const auto scores_shape_rank = scores_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          deltas_shape_rank.compatible(3),
                          "The 'input_deltas' input is expected to be a 3D. Got: ",
                          deltas_shape);
    NODE_VALIDATION_CHECK(op,
                          scores_shape_rank.compatible(3),
                          "The 'input_scores' input is expected to be a 3D. Got: ",
                          scores_shape);
    if (deltas_shape_rank.is_static() && scores_shape_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              deltas_shape[1].compatible(scores_shape[1]),
                              "Heights for inputs 'input_deltas' and 'input_scores' should be "
                              "equal. Got: ",
                              deltas_shape[1],
                              scores_shape[1]);

        NODE_VALIDATION_CHECK(op,
                              deltas_shape[2].compatible(scores_shape[2]),
                              "Width for inputs 'input_deltas' and 'input_scores' should be "
                              "equal. Got: ",
                              deltas_shape[2],
                              scores_shape[2]);
    }
    auto& rois_shape = output_shapes[0];
    auto& rois_scores_shape = output_shapes[1];

    rois_shape.resize(2);
    rois_scores_shape.resize(1);
    rois_shape[0] = post_nms_count;
    rois_shape[1] = 4;
    rois_scores_shape[0] = post_nms_count;
}

}  // namespace v6
}  // namespace op
}  // namespace ov
