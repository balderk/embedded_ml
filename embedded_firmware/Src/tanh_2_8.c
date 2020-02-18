/**
  ******************************************************************************
  * @file    tanh_2_8.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Feb 18 18:21:57 2020
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */


#include "tanh_2_8.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 5
#define AI_TOOLS_VERSION_MINOR 0
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_tanh_2_8

#undef AI_TANH_2_8_MODEL_SIGNATURE
#define AI_TANH_2_8_MODEL_SIGNATURE     "84e3a141a6b756fbcfcb9df4b7da8eb2"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue Feb 18 18:21:57 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_TANH_2_8_N_BATCHES
#define AI_TANH_2_8_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array dense_3_bias_array;   /* Array #0 */
AI_STATIC ai_array dense_3_weights_array;   /* Array #1 */
AI_STATIC ai_array dense_2_bias_array;   /* Array #2 */
AI_STATIC ai_array dense_2_weights_array;   /* Array #3 */
AI_STATIC ai_array dense_1_bias_array;   /* Array #4 */
AI_STATIC ai_array dense_1_weights_array;   /* Array #5 */
AI_STATIC ai_array dense_bias_array;   /* Array #6 */
AI_STATIC ai_array dense_weights_array;   /* Array #7 */
AI_STATIC ai_array batch_normalization_bias_array;   /* Array #8 */
AI_STATIC ai_array batch_normalization_scale_array;   /* Array #9 */
AI_STATIC ai_array input_0_output_array;   /* Array #10 */
AI_STATIC ai_array batch_normalization_output_array;   /* Array #11 */
AI_STATIC ai_array dense_output_array;   /* Array #12 */
AI_STATIC ai_array dense_nl_output_array;   /* Array #13 */
AI_STATIC ai_array dense_1_output_array;   /* Array #14 */
AI_STATIC ai_array dense_1_nl_output_array;   /* Array #15 */
AI_STATIC ai_array dense_2_output_array;   /* Array #16 */
AI_STATIC ai_array dense_2_nl_output_array;   /* Array #17 */
AI_STATIC ai_array dense_3_output_array;   /* Array #18 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor dense_3_bias;   /* Tensor #0 */
AI_STATIC ai_tensor dense_3_weights;   /* Tensor #1 */
AI_STATIC ai_tensor dense_2_bias;   /* Tensor #2 */
AI_STATIC ai_tensor dense_2_weights;   /* Tensor #3 */
AI_STATIC ai_tensor dense_1_bias;   /* Tensor #4 */
AI_STATIC ai_tensor dense_1_weights;   /* Tensor #5 */
AI_STATIC ai_tensor dense_bias;   /* Tensor #6 */
AI_STATIC ai_tensor dense_weights;   /* Tensor #7 */
AI_STATIC ai_tensor batch_normalization_bias;   /* Tensor #8 */
AI_STATIC ai_tensor batch_normalization_scale;   /* Tensor #9 */
AI_STATIC ai_tensor input_0_output;   /* Tensor #10 */
AI_STATIC ai_tensor batch_normalization_output;   /* Tensor #11 */
AI_STATIC ai_tensor dense_output;   /* Tensor #12 */
AI_STATIC ai_tensor dense_nl_output;   /* Tensor #13 */
AI_STATIC ai_tensor dense_1_output;   /* Tensor #14 */
AI_STATIC ai_tensor dense_1_nl_output;   /* Tensor #15 */
AI_STATIC ai_tensor dense_2_output;   /* Tensor #16 */
AI_STATIC ai_tensor dense_2_nl_output;   /* Tensor #17 */
AI_STATIC ai_tensor dense_3_output;   /* Tensor #18 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain batch_normalization_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain dense_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain dense_nl_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain dense_1_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain dense_1_nl_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain dense_2_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain dense_2_nl_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain dense_3_chain;   /* Chain #7 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_bn batch_normalization_layer; /* Layer #0 */
AI_STATIC ai_layer_dense dense_layer; /* Layer #1 */
AI_STATIC ai_layer_nl dense_nl_layer; /* Layer #2 */
AI_STATIC ai_layer_dense dense_1_layer; /* Layer #3 */
AI_STATIC ai_layer_nl dense_1_nl_layer; /* Layer #4 */
AI_STATIC ai_layer_dense dense_2_layer; /* Layer #5 */
AI_STATIC ai_layer_nl dense_2_nl_layer; /* Layer #6 */
AI_STATIC ai_layer_dense dense_3_layer; /* Layer #7 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
        dense_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 4,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_2_weights_array, AI_ARRAY_FORMAT_LUT4_FLOAT,
        NULL, NULL, 512,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_1_weights_array, AI_ARRAY_FORMAT_LUT4_FLOAT,
        NULL, NULL, 4096,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_weights_array, AI_ARRAY_FORMAT_LUT4_FLOAT,
        NULL, NULL, 512,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        batch_normalization_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        batch_normalization_scale_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        input_0_output_array, AI_ARRAY_FORMAT_FLOAT | AI_FMT_FLAG_IS_IO,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        batch_normalization_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_1_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_1_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_2_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_2_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        dense_3_output_array, AI_ARRAY_FORMAT_FLOAT | AI_FMT_FLAG_IS_IO,
        NULL, NULL, 4,
        AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
        dense_3_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
        1, &dense_3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_3_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 8, 4, 1, 1), AI_STRIDE_INIT(4, 4, 32, 128, 128),
        1, &dense_3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_2_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &dense_2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_2_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 64, 8, 1, 1), AI_STRIDE_INIT(4, 1, 32, 256, 256),
        1, &dense_2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_1_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &dense_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_1_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 64, 64, 1, 1), AI_STRIDE_INIT(4, 1, 32, 2048, 2048),
        1, &dense_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &dense_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 8, 64, 1, 1), AI_STRIDE_INIT(4, 1, 4, 256, 256),
        1, &dense_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        batch_normalization_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &batch_normalization_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        batch_normalization_scale, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &batch_normalization_scale_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        input_0_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &input_0_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        batch_normalization_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &batch_normalization_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &dense_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_nl_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &dense_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_1_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &dense_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_1_nl_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &dense_1_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_2_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &dense_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_2_nl_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &dense_2_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        dense_3_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
        1, &dense_3_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
        batch_normalization_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&input_0_output),
        AI_TENSOR_LIST_ENTRY(&batch_normalization_output),
        AI_TENSOR_LIST_ENTRY(&batch_normalization_scale, &batch_normalization_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        batch_normalization_layer, 0,
        BN_TYPE,
        bn, forward_bn,
        &AI_NET_OBJ_INSTANCE, &dense_layer, AI_STATIC,
        .tensors = &batch_normalization_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&batch_normalization_output),
        AI_TENSOR_LIST_ENTRY(&dense_output),
        AI_TENSOR_LIST_ENTRY(&dense_weights, &dense_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_layer, 2,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &dense_nl_layer, AI_STATIC,
        .tensors = &dense_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_nl_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&dense_output),
        AI_TENSOR_LIST_ENTRY(&dense_nl_output),
        AI_TENSOR_LIST_EMPTY,
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_nl_layer, 2,
        NL_TYPE,
        nl, forward_tanh,
        &AI_NET_OBJ_INSTANCE, &dense_1_layer, AI_STATIC,
        .tensors = &dense_nl_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_1_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&dense_nl_output),
        AI_TENSOR_LIST_ENTRY(&dense_1_output),
        AI_TENSOR_LIST_ENTRY(&dense_1_weights, &dense_1_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_1_layer, 3,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &dense_1_nl_layer, AI_STATIC,
        .tensors = &dense_1_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_1_nl_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&dense_1_output),
        AI_TENSOR_LIST_ENTRY(&dense_1_nl_output),
        AI_TENSOR_LIST_EMPTY,
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_1_nl_layer, 3,
        NL_TYPE,
        nl, forward_tanh,
        &AI_NET_OBJ_INSTANCE, &dense_2_layer, AI_STATIC,
        .tensors = &dense_1_nl_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_2_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&dense_1_nl_output),
        AI_TENSOR_LIST_ENTRY(&dense_2_output),
        AI_TENSOR_LIST_ENTRY(&dense_2_weights, &dense_2_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_2_layer, 4,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &dense_2_nl_layer, AI_STATIC,
        .tensors = &dense_2_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_2_nl_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&dense_2_output),
        AI_TENSOR_LIST_ENTRY(&dense_2_nl_output),
        AI_TENSOR_LIST_EMPTY,
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_2_nl_layer, 4,
        NL_TYPE,
        nl, forward_tanh,
        &AI_NET_OBJ_INSTANCE, &dense_3_layer, AI_STATIC,
        .tensors = &dense_2_nl_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        dense_3_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&dense_2_nl_output),
        AI_TENSOR_LIST_ENTRY(&dense_3_output),
        AI_TENSOR_LIST_ENTRY(&dense_3_weights, &dense_3_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        dense_3_layer, 5,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &dense_3_layer, AI_STATIC,
        .tensors = &dense_3_chain,
)


AI_NETWORK_OBJ_DECLARE(
        AI_NET_OBJ_INSTANCE, AI_STATIC,
        AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                           1, 1, 3504, 1,
                           NULL),
        AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                           1, 1, 512, 1,
                           NULL),
        AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_TANH_2_8_IN_NUM, &input_0_output),
        AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_TANH_2_8_OUT_NUM, &dense_3_output),
        &batch_normalization_layer, 0, NULL)


AI_DECLARE_STATIC
ai_bool tanh_2_8_configure_activations(
        ai_network *net_ctx, const ai_buffer *activation_buffer) {
    AI_ASSERT(net_ctx && activation_buffer && activation_buffer->data)

    ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
    AI_ASSERT(activations)
    AI_UNUSED(net_ctx)

    {
        /* Updating activations (byte) offsets */
        input_0_output_array.data = AI_PTR(NULL);
        input_0_output_array.data_start = AI_PTR(NULL);
        batch_normalization_output_array.data = AI_PTR(activations + 0);
        batch_normalization_output_array.data_start = AI_PTR(activations + 0);
        dense_output_array.data = AI_PTR(activations + 256);
        dense_output_array.data_start = AI_PTR(activations + 256);
        dense_nl_output_array.data = AI_PTR(activations + 256);
        dense_nl_output_array.data_start = AI_PTR(activations + 256);
        dense_1_output_array.data = AI_PTR(activations + 0);
        dense_1_output_array.data_start = AI_PTR(activations + 0);
        dense_1_nl_output_array.data = AI_PTR(activations + 0);
        dense_1_nl_output_array.data_start = AI_PTR(activations + 0);
        dense_2_output_array.data = AI_PTR(activations + 256);
        dense_2_output_array.data_start = AI_PTR(activations + 256);
        dense_2_nl_output_array.data = AI_PTR(activations + 256);
        dense_2_nl_output_array.data_start = AI_PTR(activations + 256);
        dense_3_output_array.data = AI_PTR(NULL);
        dense_3_output_array.data_start = AI_PTR(NULL);

    }
    return true;
}


AI_DECLARE_STATIC
ai_bool tanh_2_8_configure_weights(
        ai_network *net_ctx, const ai_buffer *weights_buffer) {
    AI_ASSERT(net_ctx && weights_buffer && weights_buffer->data)

    ai_ptr weights = AI_PTR(weights_buffer->data);
    AI_ASSERT(weights)
    AI_UNUSED(net_ctx)

    {
        /* Updating weights (byte) offsets */

        dense_3_bias_array.format |= AI_FMT_FLAG_CONST;
        dense_3_bias_array.data = AI_PTR(weights + 3488);
        dense_3_bias_array.data_start = AI_PTR(weights + 3488);
        dense_3_weights_array.format |= AI_FMT_FLAG_CONST;
        dense_3_weights_array.data = AI_PTR(weights + 3360);
        dense_3_weights_array.data_start = AI_PTR(weights + 3360);
        dense_2_bias_array.format |= AI_FMT_FLAG_CONST;
        dense_2_bias_array.data = AI_PTR(weights + 3328);
        dense_2_bias_array.data_start = AI_PTR(weights + 3328);
        dense_2_weights_array.format |= AI_FMT_FLAG_CONST;
        dense_2_weights_array.data = AI_PTR(weights + 3072);
        dense_2_weights_array.data_start = AI_PTR(weights + 3008);
        dense_1_bias_array.format |= AI_FMT_FLAG_CONST;
        dense_1_bias_array.data = AI_PTR(weights + 2752);
        dense_1_bias_array.data_start = AI_PTR(weights + 2752);
        dense_1_weights_array.format |= AI_FMT_FLAG_CONST;
        dense_1_weights_array.data = AI_PTR(weights + 704);
        dense_1_weights_array.data_start = AI_PTR(weights + 640);
        dense_bias_array.format |= AI_FMT_FLAG_CONST;
        dense_bias_array.data = AI_PTR(weights + 384);
        dense_bias_array.data_start = AI_PTR(weights + 384);
        dense_weights_array.format |= AI_FMT_FLAG_CONST;
        dense_weights_array.data = AI_PTR(weights + 128);
        dense_weights_array.data_start = AI_PTR(weights + 64);
        batch_normalization_bias_array.format |= AI_FMT_FLAG_CONST;
        batch_normalization_bias_array.data = AI_PTR(weights + 32);
        batch_normalization_bias_array.data_start = AI_PTR(weights + 32);
        batch_normalization_scale_array.format |= AI_FMT_FLAG_CONST;
        batch_normalization_scale_array.data = AI_PTR(weights + 0);
        batch_normalization_scale_array.data_start = AI_PTR(weights + 0);
    }

    return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_tanh_2_8_get_info(
        ai_handle network, ai_network_report *report) {
    ai_network *net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

    if (report && net_ctx) {
        ai_network_report r = {
                .model_name        = AI_TANH_2_8_MODEL_NAME,
                .model_signature   = AI_TANH_2_8_MODEL_SIGNATURE,
                .model_datetime    = AI_TOOLS_DATE_TIME,

                .compile_datetime  = AI_TOOLS_COMPILE_TIME,

                .runtime_revision  = ai_platform_runtime_get_revision(),
                .runtime_version   = ai_platform_runtime_get_version(),

                .tool_revision     = AI_TOOLS_REVISION_ID,
                .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                                      AI_TOOLS_VERSION_MICRO, 0x0},
                .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                                      AI_TOOLS_API_VERSION_MICRO, 0x0},

                .api_version            = ai_platform_api_get_version(),
                .interface_api_version  = ai_platform_interface_api_get_version(),

                .n_macc            = 6528,
                .n_inputs          = 0,
                .inputs            = NULL,
                .n_outputs         = 0,
                .outputs           = NULL,
                .activations       = AI_STRUCT_INIT,
                .params            = AI_STRUCT_INIT,
                .n_nodes           = 0,
                .signature         = 0x0,
        };

        if (!ai_platform_api_get_network_report(network, &r)) return false;

        *report = r;
        return true;
    }

    return false;
}

AI_API_ENTRY
ai_error ai_tanh_2_8_get_error(ai_handle network) {
    return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_tanh_2_8_create(
        ai_handle *network, const ai_buffer *network_config) {
    return ai_platform_network_create(
            network, network_config,
            &AI_NET_OBJ_INSTANCE,
            AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_tanh_2_8_destroy(ai_handle network) {
    return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_tanh_2_8_init(
        ai_handle network, const ai_network_params *params) {
    ai_network *net_ctx = ai_platform_network_init(network, params);
    if (!net_ctx) return false;

    ai_bool ok = true;
    ok &= tanh_2_8_configure_weights(net_ctx, &params->params);
    ok &= tanh_2_8_configure_activations(net_ctx, &params->activations);

    return ok;
}


AI_API_ENTRY
ai_i32 ai_tanh_2_8_run(
        ai_handle network, const ai_buffer *input, ai_buffer *output) {
    return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_tanh_2_8_forward(ai_handle network, const ai_buffer *input) {
    return ai_platform_network_process(network, input, NULL);
}

#undef AI_TANH_2_8_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

