#include "mpu6050.h"

#define GYRO_SCALE	0x10		// 0x00(250), 0x08(500), 0x10(1000), 0x18(2000)
#define GYRO_SCALE_FACTOR	32.8f		//131.0f, 65.5f, 32.8f, 16.4f
#define ACCEL_SCALE	0x08			//0x00(2), 0x08(4), 0x10(8), 0x18(16)
#define ACCEL_SCALE_FACTOR	8192.0f		//16384.0f, 8192.0f, 4096.0f, 2048.0f

HAL_StatusTypeDef MPU6050_Init(I2C_HandleTypeDef *hi2c) {
    uint8_t check = 0;
    uint8_t data = 0;
    HAL_StatusTypeDef status;

    // Check device ID (WHO_AM_I register should return 0x68)
    status = HAL_I2C_Mem_Read(hi2c, MPU6050_ADDR, MPU6050_REG_WHO_AM_I, 1, &check, 1, 100);
    if (status != HAL_OK || check != 0x68) {
        return HAL_ERROR; // Device not found
    }

    // Wake up MPU-6050 (Exit Sleep mode, select internal 8MHz clock)
    data = 0x00;
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_REG_PWR_MGMT_1, 1, &data, 1, 100);
    if (status != HAL_OK) return status;

    // Set Data Rate to 1kHz (SMPLRT_DIV = 7)
    data = 0x07;
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_REG_SMPLRT_DIV, 1, &data, 1, 100);
    if (status != HAL_OK) return status;

    // Set Accelerometer configuration (Full Scale ±2g)
    data = ACCEL_SCALE;
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_REG_ACCEL_CONFIG, 1, &data, 1, 100);
    if (status != HAL_OK) return status;

    // Set Gyroscope configuration (Full Scale ±250 deg/s)
    data = GYRO_SCALE;
    status = HAL_I2C_Mem_Write(hi2c, MPU6050_ADDR, MPU6050_REG_GYRO_CONFIG, 1, &data, 1, 100);
    if (status != HAL_OK) return status;

    return HAL_OK;
}

HAL_StatusTypeDef MPU6050_Read_All(I2C_HandleTypeDef *hi2c, MPU6050_t *DataStruct) {
    uint8_t Rec_Data[14];
    HAL_StatusTypeDef status;

    // Read 14 sequential registers starting from ACCEL_XOUT_H (0x3B)
    // 6 bytes Accel + 2 bytes Temp + 6 bytes Gyro
    status = HAL_I2C_Mem_Read(hi2c, MPU6050_ADDR, MPU6050_REG_ACCEL_XOUT_H, 1, Rec_Data, 14, 100);
    if (status != HAL_OK) return status;

    // Extract Raw Values
    DataStruct->Accel_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
    DataStruct->Accel_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
    DataStruct->Accel_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);
    DataStruct->Temp_RAW    = (int16_t)(Rec_Data[6] << 8 | Rec_Data[7]);
    DataStruct->Gyro_X_RAW  = (int16_t)(Rec_Data[8] << 8 | Rec_Data[9]);
    DataStruct->Gyro_Y_RAW  = (int16_t)(Rec_Data[10] << 8 | Rec_Data[11]);
    DataStruct->Gyro_Z_RAW  = (int16_t)(Rec_Data[12] << 8 | Rec_Data[13]);

    // Convert Raw Values to Physical Values:
    // Accelerometer scale factor for ±2g is 16384.0 LSB/g
    DataStruct->Ax = DataStruct->Accel_X_RAW / ACCEL_SCALE_FACTOR;
    DataStruct->Ay = DataStruct->Accel_Y_RAW / ACCEL_SCALE_FACTOR;
    DataStruct->Az = DataStruct->Accel_Z_RAW / ACCEL_SCALE_FACTOR;

    // Gyroscope scale factor for ±250 °/s is 131.0 LSB/(°/s)
    DataStruct->Gx = DataStruct->Gyro_X_RAW / GYRO_SCALE_FACTOR;
    DataStruct->Gy = DataStruct->Gyro_Y_RAW / GYRO_SCALE_FACTOR;
    DataStruct->Gz = DataStruct->Gyro_Z_RAW / GYRO_SCALE_FACTOR;

    // Temperature formula per MPU-6050 Datasheet
    DataStruct->Temperature = (float)((int16_t)DataStruct->Temp_RAW / 340.0) + 36.53f;

    return HAL_OK;
}
