// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <limits>
#include "primitive.hpp"

namespace cldnn {

/// @brief Select method for coding the prior-boxes in the @ref detection output layer.
enum class prior_box_code_type : int32_t {
    corner,
    center_size,
    corner_size
};

/// @brief Generates a list of detections based on location and confidence predictions by doing non maximum suppression.
/// @details Each row is a 7 dimension vector, which stores: [image_id, label, confidence, xmin, ymin, xmax, ymax].
/// If number of detections per image is lower than keep_top_k, will write dummy results at the end with image_id=-1.
struct detection_output : public primitive_base<detection_output> {
    CLDNN_DECLARE_PRIMITIVE(detection_output)

    /// @brief Constructs detection output primitive.
    /// @param id This primitive id.
    /// @param input_location Input location primitive id.
    /// @param input_confidence Input confidence primitive id.
    /// @param input_prior_box Input prior-box primitive id.
    /// @param num_classes Number of classes to be predicted.
    /// @param keep_top_k Number of total bounding boxes to be kept per image after NMS step.
    /// @param share_location If true bounding box are shared among different classes.
    /// @param background_label_id Background label id (-1 if there is no background class).
    /// @param nms_threshold Threshold for NMS step.
    /// @param top_k Maximum number of results to be kept in NMS.
    /// @param eta Used for adaptive NMS.
    /// @param code_type Type of coding method for bounding box.
    /// @param variance_encoded_in_target If true, variance is encoded in target; otherwise we need to adjust the predicted offset accordingly.
    /// @param confidence_threshold Only keep detections with confidences larger than this threshold.
    detection_output(const primitive_id& id,
                     const input_info& input_location,
                     const input_info& input_confidence,
                     const input_info& input_prior_box,
                     const uint32_t num_classes,
                     const uint32_t keep_top_k,
                     const bool share_location = true,
                     const int background_label_id = 0,
                     const float nms_threshold = 0.3,
                     const int top_k = -1,
                     const float eta = 1.f,
                     const prior_box_code_type code_type = prior_box_code_type::corner,
                     const bool variance_encoded_in_target = false,
                     const float confidence_threshold = -std::numeric_limits<float>::max(),
                     const int32_t prior_info_size = 4,
                     const int32_t prior_coordinates_offset = 0,
                     const bool prior_is_normalized = true,
                     const int32_t input_width = -1,
                     const int32_t input_height = -1,
                     const bool decrease_label_id = false,
                     const bool clip_before_nms = false,
                     const bool clip_after_nms = false,
                     const padding& output_padding = padding())
        : primitive_base(id, {input_location, input_confidence, input_prior_box}, {output_padding}),
          num_classes(num_classes),
          keep_top_k(keep_top_k),
          share_location(share_location),
          background_label_id(background_label_id),
          nms_threshold(nms_threshold),
          top_k(top_k),
          eta(eta),
          code_type(code_type),
          variance_encoded_in_target(variance_encoded_in_target),
          confidence_threshold(confidence_threshold),
          prior_info_size(prior_info_size),
          prior_coordinates_offset(prior_coordinates_offset),
          prior_is_normalized(prior_is_normalized),
          input_width(input_width),
          input_height(input_height),
          decrease_label_id(decrease_label_id),
          clip_before_nms(clip_before_nms),
          clip_after_nms(clip_after_nms) {
        if (decrease_label_id && background_label_id != 0)
            throw std::invalid_argument(
                "Cannot use decrease_label_id and background_label_id parameter simultaneously.");
    }

    /// @brief Number of classes to be predicted.
    const uint32_t num_classes;
    /// @brief Number of total bounding boxes to be kept per image after NMS step.
    const int keep_top_k;
    /// @brief If true, bounding box are shared among different classes.
    const bool share_location;
    /// @brief Background label id (-1 if there is no background class).
    const int background_label_id;
    /// @brief Threshold for NMS step.
    const float nms_threshold;
    /// @brief Maximum number of results to be kept in NMS.
    const int top_k;
    /// @brief Used for adaptive NMS.
    const float eta;
    /// @brief Type of coding method for bounding box.
    const prior_box_code_type code_type;
    /// @brief If true, variance is encoded in target; otherwise we need to adjust the predicted offset accordingly.
    const bool variance_encoded_in_target;
    /// @brief Only keep detections with confidences larger than this threshold.
    const float confidence_threshold;
    /// @brief Number of elements in a single prior description (4 if priors calculated using PriorBox layer, 5 - if Proposal)
    const int32_t prior_info_size;
    /// @brief Offset of the box coordinates w.r.t. the beginning of a prior info record
    const int32_t prior_coordinates_offset;
    /// @brief If true, priors are normalized to [0; 1] range.
    const bool prior_is_normalized;
    /// @brief Width of input image.
    const int32_t input_width;
    /// @brief Height of input image.
    const int32_t input_height;
    /// @brief Decrease label id to skip background label equal to 0. Can't be used simultaneously with background_label_id.
    const bool decrease_label_id;
    /// @brief Clip decoded boxes right after decoding
    const bool clip_before_nms;
    /// @brief Clip decoded boxes after nms step
    const bool clip_after_nms;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_classes);
        seed = hash_combine(seed, keep_top_k);
        seed = hash_combine(seed, share_location);
        seed = hash_combine(seed, background_label_id);
        seed = hash_combine(seed, nms_threshold);
        seed = hash_combine(seed, top_k);
        seed = hash_combine(seed, eta);
        seed = hash_combine(seed, code_type);
        seed = hash_combine(seed, variance_encoded_in_target);
        seed = hash_combine(seed, confidence_threshold);
        seed = hash_combine(seed, prior_info_size);
        seed = hash_combine(seed, prior_coordinates_offset);
        seed = hash_combine(seed, prior_is_normalized);
        seed = hash_combine(seed, input_width);
        seed = hash_combine(seed, input_height);
        seed = hash_combine(seed, decrease_label_id);
        seed = hash_combine(seed, clip_before_nms);
        seed = hash_combine(seed, clip_after_nms);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const detection_output>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(num_classes) &&
               cmp_fields(keep_top_k) &&
               cmp_fields(share_location) &&
               cmp_fields(background_label_id) &&
               cmp_fields(nms_threshold) &&
               cmp_fields(top_k) &&
               cmp_fields(eta) &&
               cmp_fields(code_type) &&
               cmp_fields(variance_encoded_in_target) &&
               cmp_fields(confidence_threshold) &&
               cmp_fields(prior_info_size) &&
               cmp_fields(prior_coordinates_offset) &&
               cmp_fields(prior_is_normalized) &&
               cmp_fields(input_width) &&
               cmp_fields(input_height) &&
               cmp_fields(decrease_label_id) &&
               cmp_fields(clip_before_nms) &&
               cmp_fields(clip_after_nms);
        #undef cmp_fields
    }

protected:
};

}  // namespace cldnn
