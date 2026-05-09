/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - Velocity Control Mode
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
#include <stdlib.h>  // 用于 atoi
#include <string.h>  // 用于 strcpy, strchr
#include <math.h>    // 用于 abs

/* USER CODE END Includes */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

// ================== 串口接收与解析隔离区 ==================
#define RX_BUF_SIZE 32
char rx_buffer[RX_BUF_SIZE];         
char process_buffer[RX_BUF_SIZE];    
uint8_t rx_index = 0;
uint8_t rx_byte;
volatile uint8_t rx_complete = 0;    

// ================== 视觉坐标与标志位 ==================
uint16_t target_x = 0;
uint16_t target_y = 0;
uint8_t  target_valid = 0;

// 【新增】：看门狗超时相关变量
uint32_t last_rx_time = 0;      // 记录最后一次收到有效坐标的时间
uint8_t  motors_stopped = 1;    // 标记电机是否已经处于停止状态（防重复下发0指令）

// ================== 闭环云台追踪控制参数 (速度环模式) ==================
#define CENTER_X 160  
#define CENTER_Y 112  
#define DEADZONE 15  

float Kp_vel_x = 0.2f;
float Kp_vel_y = 0.1f;

#define MAX_RPM 150    
#define ACCEL   15      

// ================== 控制频率(节流阀) ==================
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
  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_CAN1_Init();
  MX_USART1_UART_Init();

  /* USER CODE BEGIN 2 */
  USER_CAN1_Filter_Init();                                  
  if(HAL_CAN_Start(&hcan1) != HAL_OK) { Error_Handler(); }  
  if(HAL_CAN_ActivateNotification(&hcan1, CAN_IT_RX_FIFO0_MSG_PENDING) != HAL_OK) { Error_Handler(); } 

  // 开启单字节串口中断接收
  HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
  
  // 延时等待驱动器完全启动
  HAL_Delay(500);
  
  /* USER CODE END 2 */

  /* Infinite loop */
 /* USER CODE BEGIN WHILE */
  while (1)
  {
      // ================== 1. 安全解析坐标 & 喂狗 ==================
      if (rx_complete == 1)
      {
          rx_complete = 0; 
          
          if (process_buffer[0] == 'X') 
          {
              char *y_ptr = strchr(process_buffer, 'Y'); 
              if (y_ptr != NULL) 
              {
                  *y_ptr = '\0'; 
                  
                  target_x = atoi(&process_buffer[1]);  
                  target_y = atoi(y_ptr + 1);      
                  
                  target_valid = 1; 
                  last_rx_time = HAL_GetTick(); // 【新增】：收到坐标，重置看门狗时间
                  motors_stopped = 0;           // 【新增】：解除停止锁定
              }
          }
      }

      uint32_t current_time = HAL_GetTick();

      // ================== 2. 正常追踪控制 ==================
      if ((current_time - last_control_time >= CONTROL_INTERVAL_MS) && (target_valid == 1))
      {
          last_control_time = current_time;
          target_valid = 0; 
          
          int16_t error_x = target_x - CENTER_X;
          int16_t error_y = target_y - CENTER_Y;

			// ================== X轴计算 (1号电机) 左右旋转 ==================
          uint16_t rpm_x = 0;
          uint8_t dir_x = 0; 
          if (abs(error_x) > DEADZONE) {
              rpm_x = (uint16_t)(abs(error_x) * Kp_vel_x);
              if (rpm_x > MAX_RPM) rpm_x = MAX_RPM; 
              // 既然之前 X 方向是对的，就保持原样
              dir_x = (error_x > 0) ? 1 : 0;        
          }

          // ================== Y轴计算 (2号电机) 上下点头 ==================
          uint16_t rpm_y = 0;
          uint8_t dir_y = 0; 
          if (abs(error_y) > DEADZONE) {
              rpm_y = (uint16_t)(abs(error_y) * Kp_vel_y);
              
              
              // 最高限速压在 60，防止重力带着它冲刺坠机
              if (rpm_y >60) rpm_y = 60; 
              
              // 方向反转逻辑：如果烧进去还是只低头不抬头，就把这里的 1 : 0 换成 0 : 1
              dir_y = (error_y > 0) ? 1 : 0;  
          }
// 【核心修复】：最后的参数从 1 改成 0，代表“抛弃同步，收到立刻执行”
          Emm_V5_Vel_Control(1, dir_x, rpm_x, ACCEL, 0); 
          
          // 【物理级防堵塞】：强制延时 1 毫秒，保证 X 轴的数据发完，清空 CAN 邮箱！
          HAL_Delay(1); 
          
          // Y 轴指令单独下发，立马执行
          Emm_V5_Vel_Control(2, dir_y, rpm_y, ACCEL, 0);
      }

			
			
			
      // ================== 3. 【新增】看门狗超时急刹车 ==================
      if ((current_time - last_rx_time > 200) && (motors_stopped == 0))
      {
          // 急刹车指令同样使用独立执行 + 错峰延时，确保“刹车片”绝不失效！
          Emm_V5_Vel_Control(1, 0, 0, ACCEL, 0); 
          HAL_Delay(1);
          Emm_V5_Vel_Control(2, 0, 0, ACCEL, 0); 
          // Emm_V5_Synchronous_motion(0); // 【必须把这行删掉或者注释掉！】
          
          motors_stopped = 1; 
      }

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

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
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 168;
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
            strcpy(process_buffer, rx_buffer);
            rx_complete = 1; 
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

void Error_Handler(void)
{
  while(1);
}

#ifdef  USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
}
#endif /* USE_FULL_ASSERT */

