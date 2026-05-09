/**
  ******************************************************************************
  * File Name          : CAN.c
  * Description        : This file provides code for the configuration
  *                      of the CAN instances.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "can.h"

/* USER CODE BEGIN 0 */

__IO CAN_t can = {0};

/* USER CODE END 0 */

CAN_HandleTypeDef hcan1;

/* CAN1 init function */
void MX_CAN1_Init(void)
{

  hcan1.Instance = CAN1;
  hcan1.Init.Prescaler = 14;
  hcan1.Init.Mode = CAN_MODE_NORMAL;
  hcan1.Init.SyncJumpWidth = CAN_SJW_1TQ;
  hcan1.Init.TimeSeg1 = CAN_BS1_4TQ;
  hcan1.Init.TimeSeg2 = CAN_BS2_1TQ;
  hcan1.Init.TimeTriggeredMode = DISABLE;
  hcan1.Init.AutoBusOff = ENABLE;
  hcan1.Init.AutoWakeUp = DISABLE;
  hcan1.Init.AutoRetransmission = DISABLE;
  hcan1.Init.ReceiveFifoLocked = DISABLE;
  hcan1.Init.TransmitFifoPriority = DISABLE;
  if (HAL_CAN_Init(&hcan1) != HAL_OK)
  {
    Error_Handler();
  }

}

void HAL_CAN_MspInit(CAN_HandleTypeDef* canHandle)
{

  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(canHandle->Instance==CAN1)
  {
  /* USER CODE BEGIN CAN1_MspInit 0 */

  /* USER CODE END CAN1_MspInit 0 */
    /* CAN1 clock enable */
    __HAL_RCC_CAN1_CLK_ENABLE();

    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**CAN1 GPIO Configuration
    PB8     ------> CAN1_RX
    PB9     ------> CAN1_TX
    */
    GPIO_InitStruct.Pin = GPIO_PIN_8|GPIO_PIN_9;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF9_CAN1;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    /* CAN1 interrupt Init */
    HAL_NVIC_SetPriority(CAN1_RX0_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(CAN1_RX0_IRQn);
  /* USER CODE BEGIN CAN1_MspInit 1 */

  /* USER CODE END CAN1_MspInit 1 */
  }
}

void HAL_CAN_MspDeInit(CAN_HandleTypeDef* canHandle)
{

  if(canHandle->Instance==CAN1)
  {
  /* USER CODE BEGIN CAN1_MspDeInit 0 */

  /* USER CODE END CAN1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_CAN1_CLK_DISABLE();

    /**CAN1 GPIO Configuration
    PB8     ------> CAN1_RX
    PB9     ------> CAN1_TX
    */
    HAL_GPIO_DeInit(GPIOB, GPIO_PIN_8|GPIO_PIN_9);

    /* CAN1 interrupt Deinit */
    HAL_NVIC_DisableIRQ(CAN1_RX0_IRQn);
  /* USER CODE BEGIN CAN1_MspDeInit 1 */

  /* USER CODE END CAN1_MspDeInit 1 */
  }
}

/* USER CODE BEGIN 1 */

/**
	* @brief   ��ʼ���˲���
	* @param   ��
	* @retval  ��
	*/
void USER_CAN1_Filter_Init(void)
{
	// �������ṹ��
	CAN_FilterTypeDef  sFilterConfig;

	// ����STM32��֡ID - ��չ֡��ʽ - �������κ�����֡
	__IO uint8_t id_o, im_o; __IO uint16_t id_l, id_h, im_l, im_h;
	id_o = (0x00);
	id_h = (uint16_t)((uint16_t)id_o >> 5);								// ��3λ
	id_l = (uint16_t)((uint16_t)id_o << 11) | CAN_ID_EXT; // ��5λ
	im_o = (0x00);
	im_h = (uint16_t)((uint16_t)im_o >> 5);
	im_l = (uint16_t)((uint16_t)im_o << 11) | CAN_ID_EXT;

	// ����������
	sFilterConfig.FilterBank = 0;                      		// ������1
	sFilterConfig.FilterMode = CAN_FILTERMODE_IDMASK;  		// ����ģʽ
	sFilterConfig.FilterScale = CAN_FILTERSCALE_32BIT; 		// 32λ������λ��
	sFilterConfig.FilterIdHigh = id_h;               			// ��������ʶ���ĸ�16λֵ
	sFilterConfig.FilterIdLow = id_l;                			// ��������ʶ���ĵ�16λֵ
	sFilterConfig.FilterMaskIdHigh = im_h;           			// ���������α�ʶ���ĸ�16λֵ
	sFilterConfig.FilterMaskIdLow = im_l;            			// ���������α�ʶ���ĵ�16λֵ
	sFilterConfig.FilterFIFOAssignment = CAN_RX_FIFO0; 		// ָ���������FIFOΪ0
	sFilterConfig.FilterActivation = ENABLE;           		// ʹ�ܹ�����
	sFilterConfig.SlaveStartFilterBank = 0;           		// �ӹ��������ã�����ѡ��ӹ������ļĴ������
	
	// ���ò��Լ�
	while(HAL_CAN_ConfigFilter(&hcan1, &sFilterConfig) != HAL_OK);
}

/**
	* @brief   CAN���Ͷ���ֽ�
	* @param   ��
	* @retval  ��
	*/
void can_SendCmd(__IO uint8_t *cmd, uint8_t len)
{
	static uint32_t TxMailbox; __IO uint8_t i = 0, j = 0, k = 0, l = 0, packNum = 0;

	// ��ȥID��ַ�͹����������ݳ���
	j = len - 2;

	// ��������
	while(i < j)
	{
		// ���ݸ���
		k = j - i;

		// ��仺��
		can.CAN_TxMsg.StdId = 0x00;
		can.CAN_TxMsg.ExtId = ((uint32_t)cmd[0] << 8) | (uint32_t)packNum;
		can.txData[0] = cmd[1];
		can.CAN_TxMsg.IDE = CAN_ID_EXT;
		can.CAN_TxMsg.RTR = CAN_RTR_DATA;

		// С��8�ֽ�����
		if(k < 8)
		{
			for(l=0; l < k; l++,i++) { can.txData[l + 1] = cmd[i + 2]; } can.CAN_TxMsg.DLC = k + 1;
		}
		// ����8�ֽ�����ְ����ͣ�ÿ��������෢��8���ֽ�
		else
		{
			for(l=0; l < 7; l++,i++) { can.txData[l + 1] = cmd[i + 2]; } can.CAN_TxMsg.DLC = 8;
		}

		// ��������
		while(HAL_CAN_AddTxMessage((&hcan1), (CAN_TxHeaderTypeDef *)(&can.CAN_TxMsg), (uint8_t *)(&can.txData), (&TxMailbox)) != HAL_OK);

		// ��¼���͵ĵڼ���������
		++packNum;
	}
}

/* USER CODE END 1 */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
