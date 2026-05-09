/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
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
#include "stm32f4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define BIN1_Pin GPIO_PIN_5
#define BIN1_GPIO_Port GPIOE
#define AO_A_Pin GPIO_PIN_0
#define AO_A_GPIO_Port GPIOA
#define AO_B_Pin GPIO_PIN_1
#define AO_B_GPIO_Port GPIOA
#define ADC_Pin GPIO_PIN_4
#define ADC_GPIO_Port GPIOA
#define HW_OUT_1_Pin GPIO_PIN_5
#define HW_OUT_1_GPIO_Port GPIOA
#define HC_SR04_Echo_Pin GPIO_PIN_6
#define HC_SR04_Echo_GPIO_Port GPIOA
#define HW_OUT_2_Pin GPIO_PIN_7
#define HW_OUT_2_GPIO_Port GPIOA
#define HW_OUT_3_Pin GPIO_PIN_0
#define HW_OUT_3_GPIO_Port GPIOB
#define HW_OUT_4_Pin GPIO_PIN_1
#define HW_OUT_4_GPIO_Port GPIOB
#define KEY2_Pin GPIO_PIN_7
#define KEY2_GPIO_Port GPIOE
#define BUZZER_Pin GPIO_PIN_9
#define BUZZER_GPIO_Port GPIOE
#define PWMA_Pin GPIO_PIN_14
#define PWMA_GPIO_Port GPIOE
#define OLED_SDA_Pin GPIO_PIN_12
#define OLED_SDA_GPIO_Port GPIOB
#define AIN1_Pin GPIO_PIN_13
#define AIN1_GPIO_Port GPIOB
#define INT_6050_Pin GPIO_PIN_15
#define INT_6050_GPIO_Port GPIOB
#define PWMB_Pin GPIO_PIN_8
#define PWMB_GPIO_Port GPIOA
#define OLED_SCL_Pin GPIO_PIN_15
#define OLED_SCL_GPIO_Port GPIOA
#define KEY1_Pin GPIO_PIN_4
#define KEY1_GPIO_Port GPIOB
#define HC_SR04_Trig_Pin GPIO_PIN_5
#define HC_SR04_Trig_GPIO_Port GPIOB
#define BO_A_Pin GPIO_PIN_6
#define BO_A_GPIO_Port GPIOB
#define BO_B_Pin GPIO_PIN_7
#define BO_B_GPIO_Port GPIOB
#define SCL_6050_Pin GPIO_PIN_8
#define SCL_6050_GPIO_Port GPIOB

/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
