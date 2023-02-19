// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Mean Variance Normalization primitive.
/// @details Normalizes the input to have 0-mean and/or unit (1) variance.
struct mvn : public primitive_base<mvn> {
    CLDNN_DECLARE_PRIMITIVE(mvn)

    /// @brief Constructs mvn primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param across_channels Determines if the normalization is done across or within channels. Default is within channels.'
    /// @param normalize_variance Determines if normalize variance is applied. Default is true.
    /// @param epsilon Epsilon for not dividing by zero while normalizing.
    /// @param eps_inside_sqrt The mode of applying epsilon.
    mvn(const primitive_id& id,
        const input_info& input,
        const bool normalize_variance,
        const float epsilon,
        const bool eps_inside_sqrt,
        const bool across_channels = false,
        const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          normalize_variance(normalize_variance),
          epsilon(epsilon),
          eps_inside_sqrt(eps_inside_sqrt),
          across_channels(across_channels) {}

    /// @brief Determines if normalize variance is applied.
    bool normalize_variance;
    /// @brief Epsilon for not dividing by zero while normalizing.
    float epsilon;
    /// @brief The mode of applying epsilon.
    bool eps_inside_sqrt;
    /// @brief Determines if the normalization is done across or within channels.
    bool across_channels;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, normalize_variance);
        seed = hash_combine(seed, epsilon);
        seed = hash_combine(seed, eps_inside_sqrt);
        seed = hash_combine(seed, across_channels);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const mvn>(rhs);

        return normalize_variance == rhs_casted.normalize_variance &&
               epsilon == rhs_casted.epsilon &&
               eps_inside_sqrt == rhs_casted.eps_inside_sqrt &&
               across_channels == rhs_casted.across_channels;
    }
};
}  // namespace cldnn
