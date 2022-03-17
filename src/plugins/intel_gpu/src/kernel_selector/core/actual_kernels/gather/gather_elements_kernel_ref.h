// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_elements_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_elements_params : public base_params {
    gather_elements_params() : base_params(KernelType::GATHER_ELEMENTS), indices_rank(0), batch_dims(0), batch_merged_output(true) {}

    uint8_t indices_rank;

    uint8_t batch_dims;

    bool batch_merged_output;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gather_elements_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gather_elements_optional_params : optional_params {
    gather_elements_optional_params() : optional_params(KernelType::GATHER_ELEMENTS) {}
};

class GatherElementsKernelRef : public KernelBaseOpenCL {
public:
    GatherElementsKernelRef() : KernelBaseOpenCL("gather_elements_ref") {}
    virtual ~GatherElementsKernelRef() {}
    virtual JitConstants GetJitConstants(const gather_elements_params& params) const;
    virtual CommonDispatchData SetDefault(const gather_elements_params& params, const optional_params&) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::QUANTIZE,
                 FusedOpType::SCALE,
                 FusedOpType::ACTIVATION,
                 FusedOpType::ELTWISE };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
