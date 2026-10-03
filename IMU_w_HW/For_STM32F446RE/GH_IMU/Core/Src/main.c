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
#include <stdio.h>
#include <string.h>


/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
typedef struct
{
    int16_t accel_raw_x;
    int16_t accel_raw_y;
    int16_t accel_raw_z;

    int16_t gyro_raw_x;
    int16_t gyro_raw_y;
    int16_t gyro_raw_z;

    int16_t temperature_raw;

    float accel_x_g;
    float accel_y_g;
    float accel_z_g;

    float gyro_x_dps;
    float gyro_y_dps;
    float gyro_z_dps;

    float temperature_c;
} MPU6050_Data_t;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* HAL expects the 7-bit I2C address shifted left by one bit */
#define MPU6050_ADDR           (0x68 << 1)

#define MPU6050_REG_SMPLRT_DIV 0x19
#define MPU6050_REG_CONFIG     0x1A
#define MPU6050_REG_GYRO_CFG   0x1B
#define MPU6050_REG_ACCEL_CFG  0x1C
#define MPU6050_REG_DATA_START 0x3B
#define MPU6050_REG_PWR_MGMT_1 0x6B
#define MPU6050_REG_WHO_AM_I   0x75

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

static HAL_StatusTypeDef MPU6050_WriteRegister(uint8_t reg,
                                               uint8_t value)
{
    return HAL_I2C_Mem_Write(&hi2c1,
                             MPU6050_ADDR,
                             reg,
                             I2C_MEMADD_SIZE_8BIT,
                             &value,
                             1,
                             HAL_MAX_DELAY);
}


static HAL_StatusTypeDef MPU6050_ReadRegister(uint8_t reg,
                                              uint8_t *value)
{
    return HAL_I2C_Mem_Read(&hi2c1,
                            MPU6050_ADDR,
                            reg,
                            I2C_MEMADD_SIZE_8BIT,
                            value,
                            1,
                            HAL_MAX_DELAY);
}


static HAL_StatusTypeDef MPU6050_Init(void)
{
    uint8_t who_am_i = 0;

    /*
     * Confirm that the MPU-6050 responds.
     * WHO_AM_I should normally return 0x68.
     */
    if (MPU6050_ReadRegister(MPU6050_REG_WHO_AM_I,
                            &who_am_i) != HAL_OK)
    {
        return HAL_ERROR;
    }

    if (who_am_i != 0x68)
    {
        return HAL_ERROR;
    }

    /*
     * PWR_MGMT_1:
     * Wake up the MPU-6050 and use the X-axis gyroscope
     * clock as the internal clock source.
     */
    if (MPU6050_WriteRegister(MPU6050_REG_PWR_MGMT_1,
                              0x01) != HAL_OK)
    {
        return HAL_ERROR;
    }

    HAL_Delay(100);

    /*
     * Sample rate:
     * Gyroscope output rate = 1 kHz when DLPF is enabled.
     * Sample rate = 1000 / (1 + 9) = 100 Hz.
     */
    if (MPU6050_WriteRegister(MPU6050_REG_SMPLRT_DIV,
                              9) != HAL_OK)
    {
        return HAL_ERROR;
    }

    /*
     * Digital low-pass filter configuration.
     * DLPF_CFG = 3.
     */
    if (MPU6050_WriteRegister(MPU6050_REG_CONFIG,
                              0x03) != HAL_OK)
    {
        return HAL_ERROR;
    }

    /*
     * Gyroscope full-scale range:
     * 0x00 = +/-250 degrees per second.
     * Sensitivity = 131 LSB/(degree/second).
     */
    if (MPU6050_WriteRegister(MPU6050_REG_GYRO_CFG,
                              0x00) != HAL_OK)
    {
        return HAL_ERROR;
    }

    /*
     * Accelerometer full-scale range:
     * 0x00 = +/-2 g.
     * Sensitivity = 16384 LSB/g.
     */
    if (MPU6050_WriteRegister(MPU6050_REG_ACCEL_CFG,
                              0x00) != HAL_OK)
    {
        return HAL_ERROR;
    }

    return HAL_OK;
}


static HAL_StatusTypeDef MPU6050_ReadData(MPU6050_Data_t *data)
{
    uint8_t buffer[14];

    /*
     * Read all 14 bytes in one I2C transaction:
     *
     * 0x3B-0x40: Accelerometer
     * 0x41-0x42: Temperature
     * 0x43-0x48: Gyroscope
     */
    if (HAL_I2C_Mem_Read(&hi2c1,
                         MPU6050_ADDR,
                         MPU6050_REG_DATA_START,
                         I2C_MEMADD_SIZE_8BIT,
                         buffer,
                         sizeof(buffer),
                         HAL_MAX_DELAY) != HAL_OK)
    {
        return HAL_ERROR;
    }

    data->accel_raw_x =
        (int16_t)((buffer[0] << 8) | buffer[1]);

    data->accel_raw_y =
        (int16_t)((buffer[2] << 8) | buffer[3]);

    data->accel_raw_z =
        (int16_t)((buffer[4] << 8) | buffer[5]);

    data->temperature_raw =
        (int16_t)((buffer[6] << 8) | buffer[7]);

    data->gyro_raw_x =
        (int16_t)((buffer[8] << 8) | buffer[9]);

    data->gyro_raw_y =
        (int16_t)((buffer[10] << 8) | buffer[11]);

    data->gyro_raw_z =
        (int16_t)((buffer[12] << 8) | buffer[13]);

    /*
     * Convert raw accelerometer data to g.
     * Sensitivity for +/-2 g = 16384 LSB/g.
     */
    data->accel_x_g = data->accel_raw_x / 16384.0f;
    data->accel_y_g = data->accel_raw_y / 16384.0f;
    data->accel_z_g = data->accel_raw_z / 16384.0f;

    /*
     * Convert raw gyroscope data to degrees per second.
     * Sensitivity for +/-250 dps = 131 LSB/dps.
     */
    data->gyro_x_dps = data->gyro_raw_x / 131.0f;
    data->gyro_y_dps = data->gyro_raw_y / 131.0f;
    data->gyro_z_dps = data->gyro_raw_z / 131.0f;

    /*
     * MPU-6050 temperature conversion formula.
     */
    data->temperature_c =
        (data->temperature_raw / 340.0f) + 36.53f;

    return HAL_OK;
}


static void UART_SendString(const char *text)
{
    HAL_UART_Transmit(&huart2,
                      (uint8_t *)text,
                      strlen(text),
                      HAL_MAX_DELAY);
}

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
  MX_I2C1_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */

  MPU6050_Data_t imu_data;

  /* USER CODE BEGIN WHILE */

  UART_SendString("Scanning I2C bus...\r\n");

  HAL_StatusTypeDef result;
  uint8_t i;
  char addFound[8];
  for (i = 1; i < 128; i++) {
      // Shift address left by 1 for STM32 HAL compatibility
      result = HAL_I2C_IsDeviceReady(&hi2c1, (uint16_t)(i << 1), 3, 5);

      if (result == HAL_OK) {
          // Device found!
    	  UART_SendString("I2C device found at address: 0x02");
    	  snprintf(addFound, sizeof(addFound), "%d", i);
    	  UART_SendString("\r\n");
      }
  }
  HAL_Delay(5000); // Scan every 5 seconds

  /* USER CODE END WHILE */



  UART_SendString("\r\nMPU-6050 test starting...\r\n");

  if (HAL_I2C_IsDeviceReady(&hi2c1,
                            MPU6050_ADDR,
                            3,
                            100) != HAL_OK)
  {
      UART_SendString("MPU-6050 not detected on I2C bus.\r\n");
      UART_SendString("Check power, SDA, SCL, GND, and AD0.\r\n");

      Error_Handler();
  }

  if (MPU6050_Init() != HAL_OK)
  {
      UART_SendString("MPU-6050 initialization failed.\r\n");

      Error_Handler();
  }

  UART_SendString("MPU-6050 initialized successfully.\r\n");

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
	  char uart_buffer[200];

	  if (MPU6050_ReadData(&imu_data) == HAL_OK)
	  {
	      int length = snprintf(
	          uart_buffer,
	          sizeof(uart_buffer),
	          "ACC[g] X:%7.3f Y:%7.3f Z:%7.3f | "
	          "GYRO[dps] X:%8.2f Y:%8.2f Z:%8.2f | "
	          "TEMP:%6.2f C\r\n",
	          imu_data.accel_x_g,
	          imu_data.accel_y_g,
	          imu_data.accel_z_g,
	          imu_data.gyro_x_dps,
	          imu_data.gyro_y_dps,
	          imu_data.gyro_z_dps,
	          imu_data.temperature_c
	      );

	      if (length > 0)
	      {
	          HAL_UART_Transmit(&huart2,
	                            (uint8_t *)uart_buffer,
	                            length,
	                            HAL_MAX_DELAY);
	      }
	  }
	  else
	  {
	      UART_SendString("MPU-6050 read error.\r\n");
	  }

	  HAL_Delay(100);
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
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
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

  /** Initializes the CPU, AHB and APB buses clocks
  */
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

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
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
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
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
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
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

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
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
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
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
