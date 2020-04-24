/**
  ******************************************************************************
  * @file    nn1.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Thu Apr 23 22:56:15 2020
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


#include "nn1.h"

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
#define AI_NET_OBJ_INSTANCE g_nn1

#undef AI_NN1_MODEL_SIGNATURE
#define AI_NN1_MODEL_SIGNATURE     "9e9308ba1542fb9877e0e7e21a93c9b4"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-5.0.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Thu Apr 23 22:56:15 2020"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NN1_N_BATCHES
#define AI_NN1_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array Output_layer_bias_array;   /* Array #0 */
AI_STATIC ai_array Output_layer_weights_array;   /* Array #1 */
AI_STATIC ai_array Layer_3_bias_array;   /* Array #2 */
AI_STATIC ai_array Layer_3_weights_array;   /* Array #3 */
AI_STATIC ai_array Layer_2_bias_array;   /* Array #4 */
AI_STATIC ai_array Layer_2_weights_array;   /* Array #5 */
AI_STATIC ai_array Layer_1_bias_array;   /* Array #6 */
AI_STATIC ai_array Layer_1_weights_array;   /* Array #7 */
AI_STATIC ai_array Preproprocessing_bias_array;   /* Array #8 */
AI_STATIC ai_array Preproprocessing_scale_array;   /* Array #9 */
AI_STATIC ai_array input_0_output_array;   /* Array #10 */
AI_STATIC ai_array Preproprocessing_output_array;   /* Array #11 */
AI_STATIC ai_array Layer_1_output_array;   /* Array #12 */
AI_STATIC ai_array Layer_1_nl_output_array;   /* Array #13 */
AI_STATIC ai_array Layer_2_output_array;   /* Array #14 */
AI_STATIC ai_array Layer_2_nl_output_array;   /* Array #15 */
AI_STATIC ai_array Layer_3_output_array;   /* Array #16 */
AI_STATIC ai_array Layer_3_nl_output_array;   /* Array #17 */
AI_STATIC ai_array Output_layer_output_array;   /* Array #18 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor Output_layer_bias;   /* Tensor #0 */
AI_STATIC ai_tensor Output_layer_weights;   /* Tensor #1 */
AI_STATIC ai_tensor Layer_3_bias;   /* Tensor #2 */
AI_STATIC ai_tensor Layer_3_weights;   /* Tensor #3 */
AI_STATIC ai_tensor Layer_2_bias;   /* Tensor #4 */
AI_STATIC ai_tensor Layer_2_weights;   /* Tensor #5 */
AI_STATIC ai_tensor Layer_1_bias;   /* Tensor #6 */
AI_STATIC ai_tensor Layer_1_weights;   /* Tensor #7 */
AI_STATIC ai_tensor Preproprocessing_bias;   /* Tensor #8 */
AI_STATIC ai_tensor Preproprocessing_scale;   /* Tensor #9 */
AI_STATIC ai_tensor input_0_output;   /* Tensor #10 */
AI_STATIC ai_tensor Preproprocessing_output;   /* Tensor #11 */
AI_STATIC ai_tensor Layer_1_output;   /* Tensor #12 */
AI_STATIC ai_tensor Layer_1_nl_output;   /* Tensor #13 */
AI_STATIC ai_tensor Layer_2_output;   /* Tensor #14 */
AI_STATIC ai_tensor Layer_2_nl_output;   /* Tensor #15 */
AI_STATIC ai_tensor Layer_3_output;   /* Tensor #16 */
AI_STATIC ai_tensor Layer_3_nl_output;   /* Tensor #17 */
AI_STATIC ai_tensor Output_layer_output;   /* Tensor #18 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain Preproprocessing_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain Layer_1_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain Layer_1_nl_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain Layer_2_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain Layer_2_nl_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain Layer_3_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain Layer_3_nl_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain Output_layer_chain;   /* Chain #7 */


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_bn Preproprocessing_layer; /* Layer #0 */
AI_STATIC ai_layer_dense Layer_1_layer; /* Layer #1 */
AI_STATIC ai_layer_nl Layer_1_nl_layer; /* Layer #2 */
AI_STATIC ai_layer_dense Layer_2_layer; /* Layer #3 */
AI_STATIC ai_layer_nl Layer_2_nl_layer; /* Layer #4 */
AI_STATIC ai_layer_dense Layer_3_layer; /* Layer #5 */
AI_STATIC ai_layer_nl Layer_3_nl_layer; /* Layer #6 */
AI_STATIC ai_layer_dense Output_layer_layer; /* Layer #7 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
        Output_layer_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 4,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Output_layer_weights_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 128,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 1024,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 2048,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 512,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Preproprocessing_bias_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Preproprocessing_scale_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        input_0_output_array, AI_ARRAY_FORMAT_FLOAT | AI_FMT_FLAG_IS_IO,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Preproprocessing_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 8,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_1_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_1_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 64,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_2_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_2_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_3_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Layer_3_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
        NULL, NULL, 32,
        AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
        Output_layer_output_array, AI_ARRAY_FORMAT_FLOAT | AI_FMT_FLAG_IS_IO,
        NULL, NULL, 4,
        AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
        Output_layer_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
        1, &Output_layer_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Output_layer_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 32, 4, 1, 1), AI_STRIDE_INIT(4, 4, 128, 512, 512),
        1, &Output_layer_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_3_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
        1, &Layer_3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_3_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 32, 32, 1, 1), AI_STRIDE_INIT(4, 4, 128, 4096, 4096),
        1, &Layer_3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_2_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
        1, &Layer_2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_2_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 64, 32, 1, 1), AI_STRIDE_INIT(4, 4, 256, 8192, 8192),
        1, &Layer_2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_1_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &Layer_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_1_weights, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 8, 64, 1, 1), AI_STRIDE_INIT(4, 4, 32, 2048, 2048),
        1, &Layer_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Preproprocessing_bias, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &Preproprocessing_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Preproprocessing_scale, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &Preproprocessing_scale_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        input_0_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &input_0_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Preproprocessing_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
        1, &Preproprocessing_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_1_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &Layer_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_1_nl_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
        1, &Layer_1_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_2_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
        1, &Layer_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_2_nl_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
        1, &Layer_2_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_3_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
        1, &Layer_3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Layer_3_nl_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
        1, &Layer_3_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
        Output_layer_output, AI_STATIC,
        0x0, 0x0, AI_SHAPE_INIT(4, 1, 4, 1, 1), AI_STRIDE_INIT(4, 4, 4, 16, 16),
        1, &Output_layer_output_array, NULL)


/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
        Preproprocessing_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&input_0_output),
        AI_TENSOR_LIST_ENTRY(&Preproprocessing_output),
        AI_TENSOR_LIST_ENTRY(&Preproprocessing_scale, &Preproprocessing_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Preproprocessing_layer, 0,
        BN_TYPE,
        bn, forward_bn,
        &AI_NET_OBJ_INSTANCE, &Layer_1_layer, AI_STATIC,
        .tensors = &Preproprocessing_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Layer_1_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Preproprocessing_output),
        AI_TENSOR_LIST_ENTRY(&Layer_1_output),
        AI_TENSOR_LIST_ENTRY(&Layer_1_weights, &Layer_1_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Layer_1_layer, 2,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &Layer_1_nl_layer, AI_STATIC,
        .tensors = &Layer_1_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Layer_1_nl_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Layer_1_output),
        AI_TENSOR_LIST_ENTRY(&Layer_1_nl_output),
        AI_TENSOR_LIST_EMPTY,
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Layer_1_nl_layer, 2,
        NL_TYPE,
        nl, forward_relu,
        &AI_NET_OBJ_INSTANCE, &Layer_2_layer, AI_STATIC,
        .tensors = &Layer_1_nl_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Layer_2_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Layer_1_nl_output),
        AI_TENSOR_LIST_ENTRY(&Layer_2_output),
        AI_TENSOR_LIST_ENTRY(&Layer_2_weights, &Layer_2_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Layer_2_layer, 3,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &Layer_2_nl_layer, AI_STATIC,
        .tensors = &Layer_2_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Layer_2_nl_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Layer_2_output),
        AI_TENSOR_LIST_ENTRY(&Layer_2_nl_output),
        AI_TENSOR_LIST_EMPTY,
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Layer_2_nl_layer, 3,
        NL_TYPE,
        nl, forward_relu,
        &AI_NET_OBJ_INSTANCE, &Layer_3_layer, AI_STATIC,
        .tensors = &Layer_2_nl_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Layer_3_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Layer_2_nl_output),
        AI_TENSOR_LIST_ENTRY(&Layer_3_output),
        AI_TENSOR_LIST_ENTRY(&Layer_3_weights, &Layer_3_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Layer_3_layer, 4,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &Layer_3_nl_layer, AI_STATIC,
        .tensors = &Layer_3_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Layer_3_nl_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Layer_3_output),
        AI_TENSOR_LIST_ENTRY(&Layer_3_nl_output),
        AI_TENSOR_LIST_EMPTY,
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Layer_3_nl_layer, 4,
        NL_TYPE,
        nl, forward_relu,
        &AI_NET_OBJ_INSTANCE, &Output_layer_layer, AI_STATIC,
        .tensors = &Layer_3_nl_chain,
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
        Output_layer_chain, AI_STATIC_CONST, 4,
        AI_TENSOR_LIST_ENTRY(&Layer_3_nl_output),
        AI_TENSOR_LIST_ENTRY(&Output_layer_output),
        AI_TENSOR_LIST_ENTRY(&Output_layer_weights, &Output_layer_bias),
        AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
        Output_layer_layer, 5,
        DENSE_TYPE,
        dense, forward_dense,
        &AI_NET_OBJ_INSTANCE, &Output_layer_layer, AI_STATIC,
        .tensors = &Output_layer_chain,
)


AI_NETWORK_OBJ_DECLARE(
        AI_NET_OBJ_INSTANCE, AI_STATIC,
        AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                           1, 1, 15440, 1,
                           NULL),
        AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                           1, 1, 384, 1,
                           NULL),
        AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_NN1_IN_NUM, &input_0_output),
        AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_NN1_OUT_NUM, &Output_layer_output),
        &Preproprocessing_layer, 0, NULL)


AI_DECLARE_STATIC
ai_bool nn1_configure_activations(
        ai_network *net_ctx, const ai_buffer *activation_buffer) {
    AI_ASSERT(net_ctx && activation_buffer && activation_buffer->data)

    ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
    AI_ASSERT(activations)
    AI_UNUSED(net_ctx)

    {
        /* Updating activations (byte) offsets */
        input_0_output_array.data = AI_PTR(NULL);
        input_0_output_array.data_start = AI_PTR(NULL);
        Preproprocessing_output_array.data = AI_PTR(activations + 0);
        Preproprocessing_output_array.data_start = AI_PTR(activations + 0);
        Layer_1_output_array.data = AI_PTR(activations + 128);
        Layer_1_output_array.data_start = AI_PTR(activations + 128);
        Layer_1_nl_output_array.data = AI_PTR(activations + 128);
        Layer_1_nl_output_array.data_start = AI_PTR(activations + 128);
        Layer_2_output_array.data = AI_PTR(activations + 0);
        Layer_2_output_array.data_start = AI_PTR(activations + 0);
        Layer_2_nl_output_array.data = AI_PTR(activations + 0);
        Layer_2_nl_output_array.data_start = AI_PTR(activations + 0);
        Layer_3_output_array.data = AI_PTR(activations + 128);
        Layer_3_output_array.data_start = AI_PTR(activations + 128);
        Layer_3_nl_output_array.data = AI_PTR(activations + 128);
        Layer_3_nl_output_array.data_start = AI_PTR(activations + 128);
        Output_layer_output_array.data = AI_PTR(NULL);
        Output_layer_output_array.data_start = AI_PTR(NULL);

    }
    return true;
}


AI_DECLARE_STATIC
ai_bool nn1_configure_weights(
        ai_network *net_ctx, const ai_buffer *weights_buffer) {
    AI_ASSERT(net_ctx && weights_buffer && weights_buffer->data)

    ai_ptr weights = AI_PTR(weights_buffer->data);
    AI_ASSERT(weights)
    AI_UNUSED(net_ctx)

    {
        /* Updating weights (byte) offsets */

        Output_layer_bias_array.format |= AI_FMT_FLAG_CONST;
        Output_layer_bias_array.data = AI_PTR(weights + 15424);
        Output_layer_bias_array.data_start = AI_PTR(weights + 15424);
        Output_layer_weights_array.format |= AI_FMT_FLAG_CONST;
        Output_layer_weights_array.data = AI_PTR(weights + 14912);
        Output_layer_weights_array.data_start = AI_PTR(weights + 14912);
        Layer_3_bias_array.format |= AI_FMT_FLAG_CONST;
        Layer_3_bias_array.data = AI_PTR(weights + 14784);
        Layer_3_bias_array.data_start = AI_PTR(weights + 14784);
        Layer_3_weights_array.format |= AI_FMT_FLAG_CONST;
        Layer_3_weights_array.data = AI_PTR(weights + 10688);
        Layer_3_weights_array.data_start = AI_PTR(weights + 10688);
        Layer_2_bias_array.format |= AI_FMT_FLAG_CONST;
        Layer_2_bias_array.data = AI_PTR(weights + 10560);
        Layer_2_bias_array.data_start = AI_PTR(weights + 10560);
        Layer_2_weights_array.format |= AI_FMT_FLAG_CONST;
        Layer_2_weights_array.data = AI_PTR(weights + 2368);
        Layer_2_weights_array.data_start = AI_PTR(weights + 2368);
        Layer_1_bias_array.format |= AI_FMT_FLAG_CONST;
        Layer_1_bias_array.data = AI_PTR(weights + 2112);
        Layer_1_bias_array.data_start = AI_PTR(weights + 2112);
        Layer_1_weights_array.format |= AI_FMT_FLAG_CONST;
        Layer_1_weights_array.data = AI_PTR(weights + 64);
        Layer_1_weights_array.data_start = AI_PTR(weights + 64);
        Preproprocessing_bias_array.format |= AI_FMT_FLAG_CONST;
        Preproprocessing_bias_array.data = AI_PTR(weights + 32);
        Preproprocessing_bias_array.data_start = AI_PTR(weights + 32);
        Preproprocessing_scale_array.format |= AI_FMT_FLAG_CONST;
        Preproprocessing_scale_array.data = AI_PTR(weights + 0);
        Preproprocessing_scale_array.data_start = AI_PTR(weights + 0);
    }

    return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_nn1_get_info(
        ai_handle network, ai_network_report *report) {
    ai_network *net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

    if (report && net_ctx) {
        ai_network_report r = {
                .model_name        = AI_NN1_MODEL_NAME,
                .model_signature   = AI_NN1_MODEL_SIGNATURE,
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

                .n_macc            = 3856,
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
ai_error ai_nn1_get_error(ai_handle network) {
    return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_nn1_create(
        ai_handle *network, const ai_buffer *network_config) {
    return ai_platform_network_create(
            network, network_config,
            &AI_NET_OBJ_INSTANCE,
            AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_nn1_destroy(ai_handle network) {
    return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_nn1_init(
        ai_handle network, const ai_network_params *params) {
    ai_network *net_ctx = ai_platform_network_init(network, params);
    if (!net_ctx) return false;

    ai_bool ok = true;
    ok &= nn1_configure_weights(net_ctx, &params->params);
    ok &= nn1_configure_activations(net_ctx, &params->activations);

    return ok;
}


AI_API_ENTRY
ai_i32 ai_nn1_run(
        ai_handle network, const ai_buffer *input, ai_buffer *output) {
    return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_nn1_forward(ai_handle network, const ai_buffer *input) {
    return ai_platform_network_process(network, input, NULL);
}

#undef AI_NN1_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

