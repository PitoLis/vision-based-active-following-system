/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2024 STMicroelectronics.
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
#include "can.h"
#include "dma.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "Emm_V5.h"
#include "usart.h"   // 引入串口库
#include <stdlib.h>  // 用于 atoi 字符串转数字
#include <string.h>  // 用于 strchr 查找字符
#include <math.h>    // 用于 abs 绝对值计算

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

/* USER CODE BEGIN PV */

// ================== 串口接收缓冲与状态 ==================
#define RX_BUF_SIZE 32
char rx_buffer[RX_BUF_SIZE];
uint8_t rx_index = 0;
uint8_t rx_byte;

// ================== 视觉坐标与标志位 ==================
uint16_t target_x = 0;
uint16_t target_y = 0;
uint8_t  target_valid = 0;

// ================== 闭环云台追踪控制参数 (终极丝滑版) ==================
#define CENTER_X 160    // 160x110 画面的物理中心X
#define CENTER_Y 112    // 160x110 画面的物理中心Y
#define DEADZONE 15     // 【修改】死区略微收紧，让它粘性更强

// 【修改】因为相机帧率极高，我们将单步的倍率适度调高，保证跟手性
float Kp_step_x = 12.0f; 
float Kp_step_y = 8.0f; 

// 【核心修改】：转速大幅压低！
// 让电机像推磨一样，用缓慢的转速填满 0.05 秒的等待间隔，把离散的脉冲连成一条直线
#define RESP_RPM 30  

// ================== 控制频率(节流阀) ==================
// 【修改】主循环检测频率设为 10ms (100Hz)，这样能保证绝对不漏接 MaixCam 发来的 20Hz 数据
#define CONTROL_INTERVAL_MS 10  
uint32_t last_control_time = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

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
  MX_DMA_Init();
  MX_CAN1_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
  
  USER_CAN1_Filter_Init();                                  // 初始化CAN滤波器
  if(HAL_CAN_Start(&hcan1) != HAL_OK) { Error_Handler(); }  // 启动CAN控制器
  if(HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK) { Error_Handler(); } // 使能CAN中断

  // 开启第一次单字节串口中断接收
  HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
  
  // 延时等待驱动器完全启动
  HAL_Delay(500);
  
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
      uint32_t current_time = HAL_GetTick();

      if ((current_time - last_control_time >= CONTROL_INTERVAL_MS) && (target_valid == 1))
      {
          last_control_time = current_time;
          target_valid = 0; 
          
          int16_t error_x = target_x - CENTER_X;
          int16_t error_y = target_y - CENTER_Y;

          // X轴计算
          uint16_t step_x = 0;
          uint8_t dir_x = 0; 
          if (abs(error_x) > DEADZONE) {
              step_x = (uint16_t)(abs(error_x) * Kp_step_x);
              // 【修改】天花板设为 150，配合 100RPM 的速度，单次极限动作会很柔和
              if (step_x >1200) step_x = 1200; 
              
              dir_x = (error_x > 0) ? 1 : 0;  // 保持你调好的方向
          }

          // Y轴计算
          uint16_t step_y = 0;
          uint8_t dir_y = 0; 
          if (abs(error_y) > DEADZONE) {
              step_y = (uint16_t)(abs(error_y) * Kp_step_y);
              // 【修改】天花板设为 100
              if (step_y > 800) step_y = 800; 
              
              dir_y = (error_y > 0) ? 1 : 0;  // 保持你调好的方向
          }

          if (step_x > 0 || step_y > 0)
          {
              if(step_x > 0) Emm_V5_Pos_Control(1, dir_x, RESP_RPM, 0, step_x, 0, 1); 
              if(step_y > 0) Emm_V5_Pos_Control(2, dir_y, RESP_RPM, 0, step_y, 0, 1);
              
              Emm_V5_Synchronous_motion(0);
          }
      }

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1) 
    {
        if (rx_byte == '\n') 
        {
            rx_buffer[rx_index] = '\0'; 

            if (rx_buffer[0] == 'X') 
            {
                char *y_ptr = strchr(rx_buffer, 'Y'); 
                if (y_ptr != NULL) 
                {
                    *y_ptr = '\0'; 
                    
                    target_x = atoi(&rx_buffer[1]);  
                    target_y = atoi(y_ptr + 1);      
                    
                    target_valid = 1; 
                }
            }
            rx_index = 0; 
        }
        else if (rx_byte != '\r') 
        {
            if (rx_index < RX_BUF_SIZE - 1) 
            {
                rx_buffer[rx_index++] = rx_byte;
            }
        }
        
        HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
    }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
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
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
