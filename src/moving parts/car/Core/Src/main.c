/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
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
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "oled.h"
#include "stdio.h"
#include "motor.h"
#include "pid.h"
#include "HC_SR04.h"   
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
extern float Motor1Speed;
extern float Motor2Speed; 

extern tPid pidMotor1Speed;
extern tPid pidMotor2Speed;

extern tPid pidFollow;      // 声明外部定义的 pidFollow 变量
uint8_t OledString[20];     

extern float Mileage;

extern short Encode1Count;
extern short Encode2Count;

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
float distance = 0.0f;        // 用于存储超声波距离数据
uint8_t hc_delay_cnt = 0;     // 用于超声波测量周期的计数器
float follow_pid_out = 0.0f;  // 用于存储跟随PID的输出速度
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
  MX_TIM1_Init();
  MX_TIM2_Init();
  MX_TIM4_Init();
  MX_ADC2_Init();
  MX_USART3_UART_Init();
  MX_USART1_UART_Init();
  /* USER CODE BEGIN 2 */
    OLED_Init();             
    OLED_Clear(); 

    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1); 
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4); 
    
    HAL_TIM_Encoder_Start(&htim2, TIM_CHANNEL_ALL); 
    HAL_TIM_Encoder_Start(&htim4, TIM_CHANNEL_ALL); 
    HAL_TIM_Base_Start_IT(&htim2);                
    HAL_TIM_Base_Start_IT(&htim4);                
    
    HAL_TIM_Base_Start_IT(&htim1); 
  /* USER CODE END 2 */
   PID_init();
  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    HAL_Delay(10);
    
    // 超声波读取逻辑：保持30ms的高刷新率 (3 * 10ms)
    hc_delay_cnt++;
    if(hc_delay_cnt >= 3)
    {
        distance = HC_SR04_Read();
        hc_delay_cnt = 0; // 重置计数器
    }

    
    // ********** PID直线跟随功能 ***********
    // 修改范围：当距离在 20cm 到 50cm 之间时启动跟随
    if(distance >= 20.0f && distance <= 50.0f) 
    { 
        follow_pid_out = PID_realize(&pidFollow, distance); // PID计算输出
        
        // 对输出速度限幅
        if(follow_pid_out > 1.0f) follow_pid_out = 1.0f;
        if(follow_pid_out < -1.0f) follow_pid_out = -1.0f;
        
        motorPidSetSpeed(follow_pid_out, follow_pid_out); // 速度作用于电机上
    }
    else 
    {
        // 1. 越界直接调用最底层函数，强制PWM为0，瞬间物理刹车！
        Motor_Set(0, 0); 
        
        // 2. 清除所有PID控制器的积分残留（历史记忆），防止复位时继续蠕动
        pidMotor1Speed.err_sum = 0;
        pidMotor2Speed.err_sum = 0;
        pidFollow.err_sum = 0;
        
        // 3. 将当前误差也强行归零
        pidMotor1Speed.err = 0;
        pidMotor2Speed.err = 0;
        pidFollow.err = 0;
    }
    
    // OLED 显示刷新
    sprintf((char*)OledString, "V1:%.2f V2:%.2f", Motor1Speed, Motor2Speed);
    OLED_ShowString(0,0,OledString,12); 
    
    sprintf((char*)OledString, "Mileage:%.2f", Mileage);
    OLED_ShowString(0,1,OledString,12); 
    
    sprintf((char*)OledString, "PID:%.0f %.0f", pid_out1, pid_out2);
    OLED_ShowString(0,2,OledString,12);
    
    sprintf((char*)OledString, "E1:%d  E2:%d", Encode1Count, Encode2Count);
    OLED_ShowString(0,3,OledString,12);

    sprintf((char*)OledString, "Dist:%.2f cm    ", distance);
    OLED_ShowString(0,4,OledString,12);
        
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

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 144;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
