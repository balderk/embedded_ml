/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.h
 * @brief          : Header for main.c file.
 *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32f7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */
#define CONSERVE_POWER 0
/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */
void custom_print(const char *format, ...);
/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define MCO_Pin GPIO_PIN_0
#define MCO_GPIO_Port GPIOH
#define LD1_Pin GPIO_PIN_0
#define LD1_GPIO_Port GPIOB
#define DEBUG_4_Pin GPIO_PIN_12
#define DEBUG_4_GPIO_Port GPIOE
#define DEBUG_3_Pin GPIO_PIN_14
#define DEBUG_3_GPIO_Port GPIOE
#define DEBUG_2_Pin GPIO_PIN_15
#define DEBUG_2_GPIO_Port GPIOE
#define DEBUG_1_Pin GPIO_PIN_10
#define DEBUG_1_GPIO_Port GPIOB
#define DEBUG_0_Pin GPIO_PIN_11
#define DEBUG_0_GPIO_Port GPIOB
#define LD3_Pin GPIO_PIN_14
#define LD3_GPIO_Port GPIOB
#define STLK_RX_Pin GPIO_PIN_8
#define STLK_RX_GPIO_Port GPIOD
#define STLK_TX_Pin GPIO_PIN_9
#define STLK_TX_GPIO_Port GPIOD
#define USB_PowerSwitchOn_Pin GPIO_PIN_6
#define USB_PowerSwitchOn_GPIO_Port GPIOG
#define USB_OverCurrent_Pin GPIO_PIN_7
#define USB_OverCurrent_GPIO_Port GPIOG
#define TMS_Pin GPIO_PIN_13
#define TMS_GPIO_Port GPIOA
#define TCK_Pin GPIO_PIN_14
#define TCK_GPIO_Port GPIOA
#define SW0_Pin GPIO_PIN_3
#define SW0_GPIO_Port GPIOB
#define LD2_Pin GPIO_PIN_7
#define LD2_GPIO_Port GPIOB
/* USER CODE BEGIN Private defines */
#if CONSERVE_POWER == 1
#define SET_USER_LED(x)  ((void *) x)

#define RESET_USER_LED(x)  ((void *) x)
#else
#define SET_USER_LED(x)                                                        \
  ({ HAL_GPIO_WritePin(LD##x##_GPIO_Port, LD##x##_Pin, GPIO_PIN_SET); })

#define RESET_USER_LED(x)                                                      \
  ({ HAL_GPIO_WritePin(LD##x##_GPIO_Port, LD##x##_Pin, GPIO_PIN_RESET); })
#endif
#define SET_DEBUG_PIN(x)                                                       \
  ({ HAL_GPIO_WritePin(DEBUG_##x##_GPIO_Port, DEBUG_##x##_Pin, GPIO_PIN_SET); })

#define RESET_DEBUG_PIN(x)                                                     \
  ({                                                                           \
    HAL_GPIO_WritePin(DEBUG_##x##_GPIO_Port, DEBUG_##x##_Pin, GPIO_PIN_RESET); \
  })
/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
