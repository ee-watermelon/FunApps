#ifndef INC_MPU6050_H_
#define INC_MPU6050_H_

#include "stm32f4xx_hal.h"

// MPU6050 I2C Address (AD0 = GND -> 0x68 << 1 = 0xD0)
#define MPU6050_ADDR         (0x68 << 1)

// MPU6050 Registers
#define MPU6050_REG_SMPLRT_DIV   0x19
#define MPU6050_REG_GYRO_CONFIG  0x1B
#define MPU6050_REG_ACCEL_CONFIG 0x1C
#define MPU6050_REG_ACCEL_XOUT_H 0x3B
#define MPU6050_REG_PWR_MGMT_1   0x6B
#define MPU6050_REG_WHO_AM_I     0x75

// Data Structure
typedef struct {
    // Raw Values
    int16_t Accel_X_RAW;
    int16_t Accel_Y_RAW;
    int16_t Accel_Z_RAW;

    int16_t Gyro_X_RAW;
    int16_t Gyro_Y_RAW;
    int16_t Gyro_Z_RAW;

    int16_t Temp_RAW;

    // Scaled / Converted Values
    float Ax, Ay, Az; // Accelerometer in g
    float Gx, Gy, Gz; // Gyroscope in deg/s
    float Temperature; // Temperature in °C
} MPU6050_t;

// Function Prototypes
HAL_StatusTypeDef MPU6050_Init(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef MPU6050_Read_All(I2C_HandleTypeDef *hi2c, MPU6050_t *DataStruct);

#endif /* INC_MPU6050_H_ */
