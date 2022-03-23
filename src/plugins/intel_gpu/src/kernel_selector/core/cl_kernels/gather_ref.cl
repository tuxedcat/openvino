// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#ifdef INDEX_DIM
    #define INPUT_AXIS_INDEX (uint)(indices[indices_idx]<0?indices[indices_idx]+INDEX_DIM:indices[indices_idx])
#else
    #define INPUT_AXIS_INDEX (uint)(indices[indices_idx])
#endif

#define GET_DICTIONARY_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)
#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)

KERNEL(gather_ref)(const __global INPUT0_TYPE* dictionary,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    #if OUTPUT_LAYOUT_B_FS_YX_FSV4
        #define BSV 1
        #define FSV 4
    #elif OUTPUT_LAYOUT_B_FS_YX_FSV16
        #define BSV 1
        #define FSV 16
    #elif OUTPUT_LAYOUT_B_FS_YX_FSV32
        #define BSV 1
        #define FSV 32
    #elif OUTPUT_LAYOUT_B_FS_ZYX_FSV16
        #define BSV 1
        #define FSV 16
    #elif OUTPUT_LAYOUT_B_FS_ZYX_FSV32
        #define BSV 1
        #define FSV 32
    #elif OUTPUT_LAYOUT_BS_FS_YX_BSV16_FSV16
        #define BSV 16
        #define FSV 16
    #elif OUTPUT_LAYOUT_BFWZYX
        #define BSV 1
        #define FSV 1
    #elif OUTPUT_LAYOUT_BFZYX
        #define BSV 1
        #define FSV 1
    #elif OUTPUT_LAYOUT_BFYX
        #define BSV 1
        #define FSV 1
    #else
        printf("Not supported format.");
        //TODO: raise error
    #endif

    #if OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
        const uint x = get_global_id(0);
        const uint y = get_global_id(1);
        const uint f = get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = get_global_id(2) / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
        const uint x = get_global_id(0);
        const uint y = get_global_id(1) % OUTPUT_SIZE_Y;
        const uint z = get_global_id(1) / OUTPUT_SIZE_Y;
        const uint f = get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = get_global_id(2) / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
        const uint x = get_global_id(0) % OUTPUT_SIZE_X;
        const uint y = get_global_id(0) / OUTPUT_SIZE_X;
        const uint z = get_global_id(1) % OUTPUT_SIZE_Z;
        const uint w = get_global_id(1) / OUTPUT_SIZE_Z;
        const uint f = get_global_id(2) % OUTPUT_FEATURE_NUM;
        const uint b = get_global_id(2) / OUTPUT_FEATURE_NUM;
    #else
        printf("Not supported dimension.");
        //TODO: raise error
    #endif
    const uint indices_idx = GET_INDICES_INDEX(INDICES_INDEX_ORDER);
    const uint dictionary_idx = GET_DICTIONARY_INDEX(DICTIONARY_INDEX_ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);

    INPUT0_TYPE val = dictionary[dictionary_idx];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}