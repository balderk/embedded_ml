/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
 * All rights reserved.</center></h2>
 *
 * This software component is licensed by ST under BSD 3-Clause license,
 * the "License"; You may not use this file except in compliance with the
 * License. You may obtain a copy of the License at:
 *                        opensource.org/licenses/BSD-3-Clause
 *
 ******************************************************************************
 */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdarg.h>
#include "relu_1_none.h"
#include "sensor_data.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

CRC_HandleTypeDef hcrc;

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);

static void MX_GPIO_Init(void);


static void MX_CRC_Init(void);

/* USER CODE BEGIN PFP */
uint8_t str_buff[512];
uint8_t _str_buff[512];

float L2_norm(float *data1, float *data2, uint16_t _len) {
    float ret_val = 0;
    for (; _len > 0; _len--) {
        ret_val += powf((data1[_len - 1] - data2[_len - 1]), 2);
    }
    return sqrtf(ret_val);
}

float mae(float *data1, float *data2, uint16_t _len) {
    float ret_val = 0;
    float orig_len = (float) _len;
    if (_len == 0) {
        return 0;
    }
    for (; _len > 0; _len--) {
        ret_val += fabsf(data1[_len - 1] - data2[_len - 1]);
    }
    return ret_val / orig_len;
}


float mse(float *data1, float *data2, uint16_t _len) {
    float ret_val = 0;
    float orig_len = (float) _len;
    if (_len == 0) {
        return 0;
    }
    for (; _len > 0; _len--) {
        ret_val += powf(data1[_len - 1] - data2[_len - 1], 2);
    }
    return ret_val / orig_len;
}


static inline void handle_ai_err(ai_error err) {
    if (err.type != AI_ERROR_NONE) {
        int len = snprintf((char *) str_buff, sizeof(str_buff), "ERROR: code: %#X type; %#X", err.code, err.type);
        if (len > 0 && len < sizeof(str_buff)) {
            HAL_UART_Transmit(&huart3, str_buff, len, 100);
        } else {
            __NOP();
        }

        while (1) {
            __NOP();
        }
    }
}

void custom_print(const char *format, ...) {
    va_list arg;
    va_start(arg, format);
    int len = vsnprintf((char *) _str_buff, sizeof(_str_buff), format, arg);
    //va_end(arg);
    va_end(arg);
    __DMB();
    if (len > 0 && len < sizeof(_str_buff)) {
        HAL_UART_Transmit(&huart3, _str_buff, len, 100);
    } else {
        __NOP();
    }

}
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void) {
    /* USER CODE BEGIN 1 */

    /* USER CODE END 1 */


    /* Enable I-Cache---------------------------------------------------------*/
    SCB_EnableICache();

    /* Enable D-Cache---------------------------------------------------------*/
    SCB_EnableDCache();

    /* MCU Configuration--------------------------------------------------------*/

    /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
    HAL_Init();

    /* USER CODE BEGIN Init */
    /* USER CODE END Init */

    /* Configure the system clock */
    SystemClock_Config();

    /* USER CODE BEGIN SysInit */

    /* USER CODE END SysInit */

    /* Initialize all configured peripherals */
    MX_GPIO_Init();
    MX_CRC_Init();
    MX_X_CUBE_AI_Init();
    /* USER CODE BEGIN 2 */

    MX_USART3_UART_Init();

    /* USER CODE END 2 */



    /* Infinite loop */
    /* USER CODE BEGIN WHILE */

    AI_ALIGNED(4)
    static ai_u8 relu_1_activations[AI_RELU_1_NONE_DATA_ACTIVATIONS_SIZE];
    static ai_handle relu_1_none_handle = AI_HANDLE_NULL;
    const ai_network_params relu_1_params = {
            AI_RELU_1_NONE_DATA_WEIGHTS(ai_relu_1_none_data_weights_get()),
            AI_RELU_1_NONE_DATA_ACTIVATIONS(relu_1_activations)};

    AI_ALIGNED(4)
    static ai_u8 relu_2_activations[AI_RELU_1_NONE_DATA_ACTIVATIONS_SIZE];
    static ai_handle relu_1_8_handle = AI_HANDLE_NULL;
    const ai_network_params relu_2_params = {
            AI_RELU_1_NONE_DATA_WEIGHTS(ai_relu_1_none_data_weights_get()),
            AI_RELU_1_NONE_DATA_ACTIVATIONS(relu_1_activations)};


    ai_error err = ai_relu_1_none_create(&relu_1_none_handle, AI_RELU_1_NONE_DATA_CONFIG);
    handle_ai_err(err);

    err = ai_relu_1_none_create(&relu_1_8_handle, AI_RELU_1_8_DATA_CONFIG);
    handle_ai_err(err);



    /* 3 - Initialize the NN - Ready to be used */
    if (!ai_relu_1_none_init(relu_1_none_handle, &relu_1_params)) {
        err = ai_relu_1_none_get_error(relu_1_none_handle);
        ai_relu_1_none_destroy(relu_1_none_handle);
        relu_1_none_handle = AI_HANDLE_NULL;
        handle_ai_err(err);
    }
    /* 3 - Initialize the NN - Ready to be used */
    if (!ai_relu_1_8_init(relu_1_none_handle, &relu_1_params)) {
        err = ai_relu_1_none_get_error(relu_1_none_handle);
        ai_relu_1_none_destroy(relu_1_none_handle);
        relu_1_none_handle = AI_HANDLE_NULL;
        handle_ai_err(err);
    }

    AI_ALIGNED(4)
    float target_data[AI_RELU_1_NONE_OUT_1_SIZE];

    AI_ALIGNED(4)
    static ai_i8 in_data[AI_RELU_1_NONE_IN_1_SIZE_BYTES];

    AI_ALIGNED(4)
    static ai_i8 out_data_1[AI_RELU_1_NONE_OUT_1_SIZE_BYTES];
    AI_ALIGNED(4)
    static ai_i8 out_data_2[AI_RELU_1_NONE_OUT_1_SIZE_BYTES];

    static float *output_data_1 = (float *) out_data_1;
    static float *output_data_2 = (float *) out_data_2;
    static float *input_data = (float *) in_data;
    ai_buffer inputs[] = AI_RELU_1_NONE_IN;
    ai_buffer outputs_1[] = AI_RELU_1_NONE_OUT;
    ai_buffer outputs_2[] = AI_RELU_1_8_OUT;
    ai_buffer *input = &inputs[0];
    ai_buffer *output_1 = &outputs_1[0];
    ai_buffer *output_2 = &outputs_2[0];
    input->data = in_data;
    output_1->data = out_data_1;
    output_2->data = out_data_2;
    sensor_data_source_t type = SENSOR_DATA_TRAIN;
    int n_batch;
    while (1) {
        RESET_DEBUG_PIN(2);
        SET_USER_LED(1);
        SET_DEBUG_PIN(0);
        if (type == SENSOR_DATA_TRAIN) {
            type = SENSOR_DATA_TEST;
            custom_print((char *) "%s:\n", "TEST");
        } else {
            type = SENSOR_DATA_TRAIN;
            custom_print((char *) "%s:\n", "TRAIN");
        }
        new_sensor_reading(type);
        get_sensor_reading(input_data);
        get_sensor_values(target_data);
        RESET_DEBUG_PIN(0);
        SET_DEBUG_PIN(1);
        n_batch = ai_relu_1_none_run(relu_1_none_handle, input, output_1);
        RESET_DEBUG_PIN(1);
        if (n_batch != 1) {
            err = ai_relu_1_none_get_error(relu_1_none_handle);
            handle_ai_err(err);
        }


        SET_DEBUG_PIN(2);
        n_batch = ai_relu_1_none_run(relu_1_8_handle, input, output_2);
        RESET_DEBUG_PIN(2);
        if (n_batch != 1) {
            err = ai_relu_1_none_get_error(relu_1_8_handle);
            handle_ai_err(err);
        }

        custom_print("input data:\t");
        for (int i = 0; i < AI_RELU_1_NONE_IN_1_SIZE; i++) {
            int len = snprintf(
                    (char *) str_buff, sizeof(str_buff), "%4d.%.2d ", (int) input_data[i],
                    abs(((int) (input_data[i] * 100)) - ((int) input_data[i]) * 100));
            __DMB();
            if (len > 0 && len < sizeof(str_buff)) {
                custom_print((char *) str_buff);
            } else {
                __NOP();
            }
        }
        custom_print("\ntarget:  \t");
        for (int i = 0; i < AI_RELU_1_NONE_OUT_1_SIZE; i++) {
            int len = snprintf(
                    (char *) str_buff, sizeof(str_buff), "%4d.%.2d ", (int) target_data[i],
                    abs(((int) (target_data[i] * 100)) - ((int) target_data[i]) * 100));
            __DMB();
            if (len > 0 && len < sizeof(str_buff)) {
                custom_print((char *) str_buff);
            } else {
                __NOP();
            }
        }
        custom_print("\noutput 1:\t");
        for (int i = 0; i < AI_RELU_1_NONE_OUT_1_SIZE; i++) {
            int len = snprintf(
                    (char *) str_buff, sizeof(str_buff), "%4d.%.2d ", (int) output_data_1[i],
                    abs(((int) (output_data_1[i] * 100)) - ((int) output_data_1[i]) * 100));
            __DMB();
            if (len > 0 && len < sizeof(str_buff)) {
                custom_print((char *) str_buff);
            } else {
                __NOP();
            }
        }

        custom_print("\noutput 2:\t");
        for (int i = 0; i < AI_RELU_1_NONE_OUT_1_SIZE; i++) {
            int len = snprintf(
                    (char *) str_buff, sizeof(str_buff), "%4d.%.2d ", (int) output_data_2[i],
                    abs(((int) (output_data_2[i] * 100)) - ((int) output_data_2[i]) * 100));
            __DMB();
            if (len > 0 && len < sizeof(str_buff)) {
                custom_print((char *) str_buff);
            } else {
                __NOP();
            }
        }

        float mae_1 = mae(target_data, output_data_1, AI_RELU_1_NONE_OUT_1_SIZE);
        float mae_2 = mae(target_data, output_data_2, AI_RELU_1_NONE_OUT_1_SIZE);
        float mae_3 = mae(output_data_2, output_data_1, AI_RELU_1_NONE_OUT_1_SIZE);

        custom_print("\nmae t - 1: %4d.%.2d \n",
                     (int) mae_1, (abs(((int) (mae_1 * 100)) - ((int) mae_1) * 100)));

        custom_print("mae t - 2: %4d.%.2d \n",
                     (int) mae_2, abs(((int) (mae_2 * 100)) - ((int) mae_2) * 100));

        custom_print("mae 1 - 2: %4d.%.2d \n",
                     (int) mae_3, abs(((int) (mae_3 * 100)) - ((int) mae_3) * 100));

        float mse_1 = L2_norm(target_data, output_data_1, AI_RELU_1_NONE_OUT_1_SIZE);
        float mse_2 = L2_norm(target_data, output_data_2, AI_RELU_1_NONE_OUT_1_SIZE);
        float mse_3 = L2_norm(output_data_2, output_data_1, AI_RELU_1_NONE_OUT_1_SIZE);

        custom_print("mse t - 1: %4d.%.2d \n",
                     (int) mse_1, (abs(((int) (mse_1 * 100)) - ((int) mse_1) * 100)));

        custom_print("mse t - 2: %4d.%.2d \n",
                     (int) mse_2, abs(((int) (mse_2 * 100)) - ((int) mse_2) * 100));

        custom_print("mse 1 - 2: %4d.%.2d \n",
                     (int) mse_3, abs(((int) (mse_3 * 100)) - ((int) mse_3) * 100));

        custom_print("\n");

        HAL_Delay(10);
        SET_USER_LED(2);
        SET_USER_LED(3);

        HAL_Delay(10);
        RESET_USER_LED(3);
        RESET_USER_LED(2);
        RESET_USER_LED(1);

        /* USER CODE END WHILE */

        MX_X_CUBE_AI_Process();
        /* USER CODE BEGIN 3 */
    }
    /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void) {
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

    /** Configure LSE Drive Capability
    */
    HAL_PWR_EnableBkUpAccess();
    /** Configure the main internal regulator output voltage
    */
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    /** Initializes the CPU, AHB and APB busses clocks
    */
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLM = 4;
    RCC_OscInitStruct.PLL.PLLN = 216;
    RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
    RCC_OscInitStruct.PLL.PLLQ = 9;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
        Error_Handler();
    }
    /** Activate the Over-Drive mode
    */
    if (HAL_PWREx_EnableOverDrive() != HAL_OK) {
        Error_Handler();
    }
    /** Initializes the CPU, AHB and APB busses clocks
    */
    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
                                  | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK) {
        Error_Handler();
    }
    PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART3;
    PeriphClkInitStruct.Usart3ClockSelection = RCC_USART3CLKSOURCE_PCLK1;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
        Error_Handler();
    }
}

/**
  * @brief CRC Initialization Function
  * @param None
  * @retval None
  */
static void MX_CRC_Init(void) {

    /* USER CODE BEGIN CRC_Init 0 */

    /* USER CODE END CRC_Init 0 */

    /* USER CODE BEGIN CRC_Init 1 */

    /* USER CODE END CRC_Init 1 */
    hcrc.Instance = CRC;
    hcrc.Init.DefaultPolynomialUse = DEFAULT_POLYNOMIAL_ENABLE;
    hcrc.Init.DefaultInitValueUse = DEFAULT_INIT_VALUE_ENABLE;
    hcrc.Init.InputDataInversionMode = CRC_INPUTDATA_INVERSION_NONE;
    hcrc.Init.OutputDataInversionMode = CRC_OUTPUTDATA_INVERSION_DISABLE;
    hcrc.InputDataFormat = CRC_INPUTDATA_FORMAT_BYTES;
    if (HAL_CRC_Init(&hcrc) != HAL_OK) {
        Error_Handler();
    }
    /* USER CODE BEGIN CRC_Init 2 */

    /* USER CODE END CRC_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART3_UART_Init(void) {

    /* USER CODE BEGIN USART3_Init 0 */

    /* USER CODE END USART3_Init 0 */

    /* USER CODE BEGIN USART3_Init 1 */

    /* USER CODE END USART3_Init 1 */
    huart3.Instance = USART3;
    huart3.Init.BaudRate = 115200;
    huart3.Init.WordLength = UART_WORDLENGTH_8B;
    huart3.Init.StopBits = UART_STOPBITS_1;
    huart3.Init.Parity = UART_PARITY_NONE;
    huart3.Init.Mode = UART_MODE_TX_RX;
    huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart3.Init.OverSampling = UART_OVERSAMPLING_16;
    huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart3) != HAL_OK) {
        Error_Handler();
    }
    /* USER CODE BEGIN USART3_Init 2 */

    /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void) {
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    /* GPIO Ports Clock Enable */
    __HAL_RCC_GPIOC_CLK_ENABLE();
    __HAL_RCC_GPIOH_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOE_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOG_CLK_ENABLE();

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(GPIOB, LD1_Pin | DEBUG_1_Pin | DEBUG_0_Pin | LD3_Pin
                             | LD2_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(GPIOE, DEBUG_4_Pin | DEBUG_3_Pin | DEBUG_2_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin Output Level */
    HAL_GPIO_WritePin(USB_PowerSwitchOn_GPIO_Port, USB_PowerSwitchOn_Pin, GPIO_PIN_RESET);

    /*Configure GPIO pin : USER_Btn_Pin */
    GPIO_InitStruct.Pin = USER_Btn_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pins : RMII_MDC_Pin RMII_RXD0_Pin RMII_RXD1_Pin */
    GPIO_InitStruct.Pin = RMII_MDC_Pin | RMII_RXD0_Pin | RMII_RXD1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /*Configure GPIO pins : RMII_REF_CLK_Pin RMII_MDIO_Pin RMII_CRS_DV_Pin */
    GPIO_InitStruct.Pin = RMII_REF_CLK_Pin | RMII_MDIO_Pin | RMII_CRS_DV_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /*Configure GPIO pins : LD1_Pin LD3_Pin LD2_Pin */
    GPIO_InitStruct.Pin = LD1_Pin | LD3_Pin | LD2_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /*Configure GPIO pins : DEBUG_4_Pin DEBUG_3_Pin DEBUG_2_Pin */
    GPIO_InitStruct.Pin = DEBUG_4_Pin | DEBUG_3_Pin | DEBUG_2_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

    /*Configure GPIO pins : DEBUG_1_Pin DEBUG_0_Pin */
    GPIO_InitStruct.Pin = DEBUG_1_Pin | DEBUG_0_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /*Configure GPIO pin : RMII_TXD1_Pin */
    GPIO_InitStruct.Pin = RMII_TXD1_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
    HAL_GPIO_Init(RMII_TXD1_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : USB_PowerSwitchOn_Pin */
    GPIO_InitStruct.Pin = USB_PowerSwitchOn_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(USB_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pin : USB_OverCurrent_Pin */
    GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pins : USB_SOF_Pin USB_ID_Pin USB_DM_Pin USB_DP_Pin */
    GPIO_InitStruct.Pin = USB_SOF_Pin | USB_ID_Pin | USB_DM_Pin | USB_DP_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /*Configure GPIO pin : USB_VBUS_Pin */
    GPIO_InitStruct.Pin = USB_VBUS_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(USB_VBUS_GPIO_Port, &GPIO_InitStruct);

    /*Configure GPIO pins : RMII_TX_EN_Pin RMII_TXD0_Pin */
    GPIO_InitStruct.Pin = RMII_TX_EN_Pin | RMII_TXD0_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
    HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void) {
    /* USER CODE BEGIN Error_Handler_Debug */
    /* User can add his own implementation to report the HAL error return state */

    /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{ 
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line
     number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line)
   */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
