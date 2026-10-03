################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (14.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/CMSIS/RTOS2/Source/os_systick.c 

OBJS += \
./Drivers/CMSIS/RTOS2/Source/os_systick.o 

C_DEPS += \
./Drivers/CMSIS/RTOS2/Source/os_systick.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/CMSIS/RTOS2/Source/%.o Drivers/CMSIS/RTOS2/Source/%.su Drivers/CMSIS/RTOS2/Source/%.cyclo: ../Drivers/CMSIS/RTOS2/Source/%.c Drivers/CMSIS/RTOS2/Source/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F446xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/RTOS2/Include" -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/NN/Include" -I../Drivers/CMSIS/Include -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers" -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/DSP/Include" -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/Core_A/Include" -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/DSP/PrivateInclude" -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/DAP/Firmware/Config" -I"C:/Users/suzie/STM32CubeIDE/workspace_2.2.0/IMU_1/Drivers/CMSIS/DSP/ComputeLibrary/Include" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-CMSIS-2f-RTOS2-2f-Source

clean-Drivers-2f-CMSIS-2f-RTOS2-2f-Source:
	-$(RM) ./Drivers/CMSIS/RTOS2/Source/os_systick.cyclo ./Drivers/CMSIS/RTOS2/Source/os_systick.d ./Drivers/CMSIS/RTOS2/Source/os_systick.o ./Drivers/CMSIS/RTOS2/Source/os_systick.su

.PHONY: clean-Drivers-2f-CMSIS-2f-RTOS2-2f-Source

