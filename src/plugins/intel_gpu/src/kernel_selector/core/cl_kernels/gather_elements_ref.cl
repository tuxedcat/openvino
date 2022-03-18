// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_UPDATES_INDEX(prefix, idx_order) CAT(prefix, _GET_INDEX)(idx_order)
#define GET_OUTPUT_INDEX(out_order) OUTPUT_GET_INDEX(out_order)

#define INDICES_MAX_DIM 6

KERNEL(gather_elements_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

    //TODO: Calculate Index or INPUT0_GET_INDEX(b,f,y,x)같은거 사용?
    const uint in0b=INPUT0_BATCH_NUM;
    const uint in0f=INPUT0_FEATURE_NUM;
    const uint in0y=INPUT0_SIZE_Y;
    const uint in0x=INPUT0_SIZE_X;
    const uint in1b=INPUT1_BATCH_NUM;
    const uint in1f=INPUT1_FEATURE_NUM;
    const uint in1y=INPUT1_SIZE_Y;
    const uint in1x=INPUT1_SIZE_X;

printf("%d %d %d %d\n",in0b,in0f,in0y,in0x);
printf("%d %d %d %d\n",in1b,in1f,in1y,in1x);
//    for(uint b=0;b<in0b;b++){
//        for(uint f=0;f<in0f;f++){
//            for(uint y=0;y<in0y;y++){
//                for(uint x=0;x<in0x;x++){
//                    printf("%f ",data[INPUT0_GET_INDEX(b,f,y,x)]);
//                }
//            }
//        }
//    }
//    for(uint b=0;b<in1b;b++){
//        for(uint f=0;f<in1f;f++){
//            for(uint y=0;y<in1y;y++){
//                for(uint x=0;x<in1x;x++){
//                    printf("%d ",indices[INPUT1_GET_INDEX(b,f,y,x)]);
//                }
//            }
//        }
//    }

#if AXIS==0
    #define AXIS_LEN0 INPUT0_BATCH_NUM
    #define AXIS_LEN1 INPUT1_BATCH_NUM
#elif AXIS==1
    #define AXIS_LEN0 INPUT0_FEATURE_NUM
    #define AXIS_LEN1 INPUT1_FEATURE_NUM
#elif AXIS==2
    #define AXIS_LEN0 INPUT0_SIZE_Y
    #define AXIS_LEN1 INPUT1_SIZE_Y
#else
    #define AXIS_LEN0 INPUT0_SIZE_X
    #define AXIS_LEN1 INPUT1_SIZE_X
#endif

    for(uint b=0;b<in1b;b++){
        for(uint f=0;f<in1f;f++){
            for(uint y=0;y<in1y;y++){
                for(uint x=0;x<in1x;x++){
                    int axis_val=indices[INPUT1_GET_INDEX(b,f,y,x)];
                    if(axis_val<0)
                        axis_val+=AXIS_LEN0;
                    #if AXIS==0
                        if(dim0<=4 && dim1<=4 && dim2<=4){
                            printf("0 %d %d %d %d %d\n",b,f,y,x,axis_val);
                            printf("1 %f\n",data[INPUT0_GET_INDEX(axis_val,f,y,x)]);
                            printf("2 %d\n",INPUT0_GET_INDEX(axis_val,f,y,x));
                            printf("\n");
                        }
                        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(axis_val,f,y,x)];
                    #elif AXIS==1
                        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(b,axis_val,y,x)];
                    #elif AXIS==2
                        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(b,f,axis_val,x)];
                    #else
                        if(dim0<=4 && dim1<=4 && dim2<=4){
                            printf("0 %d %d %d %d %d\n",b,f,y,x,axis_val);
                            printf("1 %f\n",data[INPUT0_GET_INDEX(b,f,y,axis_val)]);
                            printf("2 %d\n",INPUT0_GET_INDEX(b,f,y,axis_val));
                            printf("\n");
                        }
                        output[OUTPUT_GET_INDEX(b,f,y,x)]=data[INPUT0_GET_INDEX(b,f,y,axis_val)];
                    #endif
                }
            }
        }
    }

    if(!dim0 && !dim1 && !dim2)
        printf("hello?\n");
    
    // Copy data to output as slice size
    #if HAS_FUSED_OPS
        #if OUTPUT_DIMS == 4
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint f_pitch = y_pitch * OUTPUT_SIZE_Y;
        #elif OUTPUT_DIMS == 5
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint f_pitch = z_pitch * OUTPUT_SIZE_Z;
        #else
            const uint y_pitch = OUTPUT_SIZE_X;
            const uint z_pitch = y_pitch * OUTPUT_SIZE_Y;
            const uint w_pitch = z_pitch * OUTPUT_SIZE_Z;
            const uint f_pitch = w_pitch * OUTPUT_SIZE_W;
        #endif
        const uint b_pitch = f_pitch * OUTPUT_FEATURE_NUM;
    #endif
}

#undef INDICES_MAX_DIM
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef OUT_ORDER
#undef IDX_ORDER
#undef IN_ORDER
