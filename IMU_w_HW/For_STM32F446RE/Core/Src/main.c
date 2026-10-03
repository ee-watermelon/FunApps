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
#include <main.h>
#include <stdio.h>
#include <string.h>
#include "mpu6050.h"
#include <stdint.h>

#define MAX_RETRY_ATTEMPTS 3

I2C_HandleTypeDef hi2c1;
UART_HandleTypeDef huart2;
MPU6050_t myMPU6050;

typedef struct __attribute__((packed)) {
    uint8_t  header[2];  // 0xAA, 0x55 (Sync header)
    float    accel_x;    // 4 bytes
    float    accel_y;    // 4 bytes
    float    accel_z;    // 4 bytes
    float    gyro_x;     // 4 bytes (deg/s)
    float    gyro_y;     // 4 bytes (deg/s)
    float    gyro_z;     // 4 bytes (deg/s)
    float    temp;       // 4 bytes (°C)
} ImuPacket;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART2_UART_Init(void);
void Initialize_MPU6050_With_Recovery(void);
void I2C_Bus_Recovery_Routine(void);
void UART_formatData(MPU6050_t dataIn, ImuPacket *dataOut);

static void UART_SendString(const char *text)
{
    HAL_UART_Transmit(&huart2,
                      (uint8_t *)text,
                      strlen(text),
                      HAL_MAX_DELAY);
}

ImuPacket packet;

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_USART2_UART_Init();

  Initialize_MPU6050_With_Recovery();

  while (1)
  {
	if (MPU6050_Read_All(&hi2c1, &myMPU6050) == HAL_OK)
	  {
		  packet.header[0] = 0xAA;
		  packet.header[1] = 0x55;
		  packet.accel_x = myMPU6050.Ax;
		  packet.accel_y = myMPU6050.Ay;
		  packet.accel_z = myMPU6050.Az;
		  packet.gyro_x = myMPU6050.Gx;
		  packet.gyro_y = myMPU6050.Gy;
		  packet.gyro_z = myMPU6050.Gz;
		  packet.temp = myMPU6050.Temperature;

		  int length = sizeof(ImuPacket);
	      if (length > 0)
	      {
	          HAL_UART_Transmit(&huart2,
	                            (uint8_t*)&packet,
	                            length,
	                            HAL_MAX_DELAY);
	      }
	  }
	  else
	  {
	      UART_SendString("MPU-6050 read error.\r\n");
	  }

	  HAL_Delay(20);
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
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

static void MX_I2C1_Init(void)
{
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

void I2C_Bus_Recovery_Routine(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    HAL_I2C_DeInit(&hi2c1);
    __HAL_RCC_GPIOB_CLK_ENABLE();

    GPIO_InitStruct.Pin = GPIO_PIN_8 | GPIO_PIN_9; // SCL and SDA
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_OD;    // Open-drain output
    GPIO_InitStruct.Pull = GPIO_NOPULL;           // Rely on external/internal pull-ups
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // Ensure both lines are released HIGH initially
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_9, GPIO_PIN_SET);
    HAL_Delay(1);

    if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_RESET)
    {
        for (int i = 0; i < 9; i++)
        {
            // Toggle SCL LOW then HIGH
            HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET);
            for (volatile int j = 0; j < 1000; j++) __NOP(); // Short delay (~10-50us)

            HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET);
            for (volatile int j = 0; j < 1000; j++) __NOP();

            // Check if slave has released SDA yet
            if (HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_9) == GPIO_PIN_SET) {
                break; // Bus is free! Exit loop early.
            }
        }
    }

    MX_I2C1_Init();
}

void Initialize_MPU6050_With_Recovery(void)
{
    uint8_t attempts = 0;

    while (attempts < MAX_RETRY_ATTEMPTS)
    {
        // Test if device acknowledges
        if (HAL_I2C_IsDeviceReady(&hi2c1, MPU6050_ADDR, 2, 50) == HAL_OK)
        {
            if (MPU6050_Init(&hi2c1) == HAL_OK) {
                UART_SendString("MPU-6050 initialized successfully.\r\n");
                return; // EXIT FUNCTION ON SUCCESS!
            }
        }

        // If it failed, increment attempt counter and run cleanup
        attempts++;

        char debug_msg[64];
        snprintf(debug_msg, sizeof(debug_msg),
                           "[WARNING] Attempt %d/%d failed. Recovering...\r\n",
                           attempts, MAX_RETRY_ATTEMPTS);
        UART_SendString(debug_msg);

        I2C_Bus_Recovery_Routine();
        HAL_Delay(500); // Cool down before retrying
    }

    // If we exhausted all attempts, halt or handle critical failure
    UART_SendString("[CRITICAL] MPU6050 initialization failed permanently.\r\n");
    Error_Handler();
}

void Error_Handler(void)
{
	HAL_Delay(1000);
}
#ifdef USE_FULL_ASSERT
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
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
