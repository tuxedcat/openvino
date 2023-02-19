// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief Normalizes results so they sum to 1.
/// @details
/// @par Algorithm:
/// @par Where:
struct region_yolo : public primitive_base<region_yolo> {
    CLDNN_DECLARE_PRIMITIVE(region_yolo)

    /// @brief Constructs region_yolo primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param dimension Defines a scope of normalization (see #dimension).
    region_yolo(const primitive_id& id,
                const input_info& input,
                const uint32_t coords,
                const uint32_t classes,
                const uint32_t num,
                const uint32_t mask_size = 0,
                const bool do_softmax = true,
                const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          coords(coords),
          classes(classes),
          num(num),
          mask_size(mask_size),
          do_softmax(do_softmax) {}

    /// @brief Defines a scope of a region yolo normalization
    /// @details
    /// Specific behaviour is determined by these parameters, as follows:
    uint32_t coords;
    uint32_t classes;
    uint32_t num;
    uint32_t mask_size;
    bool do_softmax;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, coords);
        seed = hash_combine(seed, classes);
        seed = hash_combine(seed, num);
        seed = hash_combine(seed, mask_size);
        seed = hash_combine(seed, do_softmax);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const region_yolo>(rhs);

        return coords == rhs_casted.coords &&
               classes == rhs_casted.classes &&
               num == rhs_casted.num &&
               mask_size == rhs_casted.mask_size &&
               do_softmax == rhs_casted.do_softmax;
    }
};
}  // namespace cldnn
#pragma once
