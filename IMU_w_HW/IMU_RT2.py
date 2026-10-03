from math import pi
import struct
import time
import threading
import queue
import serial
import numpy as np
from vpython import *
#from vpython import scene, box, vector, color, rate
from ahrs.filters import Madgwick
import math

# Packet definition:
# '<' = Little-endian
# 'BB' = 2 bytes (Header 0xAA, 0x55)
# '7f' = 7 32-bit floats (ax, ay, az, gx, gy, gz, temp)
PACKET_FORMAT = '<BB7f'
PACKET_SIZE = struct.calcsize(PACKET_FORMAT)  # Exact 30 bytes
HEADER = b'\xAA\x55'
PORT = "COM27"
printOut = ""

################# Calibration ########################
CALIBRATION = 0
CAL_SAMPLE_NUM = 200
######################################################

data_queue = queue.Queue(maxsize=1)

def read_one_binary_packet(ser, raw_buffer):
    
    while True:
        if ser.in_waiting > 0:
            raw_buffer.extend(ser.read(ser.in_waiting))
        else:
            time.sleep(0.001)
            
        while len(raw_buffer) >= PACKET_SIZE:
            idx = raw_buffer.find(HEADER)
            if idx == -1:
                raw_buffer = raw_buffer[-1:]
                break
            if idx > 0:
                del raw_buffer[:idx]
            if len(raw_buffer) < PACKET_SIZE:
                break
                
            packet_bytes = bytes(raw_buffer[:PACKET_SIZE])
            del raw_buffer[:PACKET_SIZE]
            
            _, _, ax, ay, az, gx, gy, gz, temp = struct.unpack(PACKET_FORMAT, packet_bytes)
            accel = np.array([ax, ay, az], dtype=float)
            gyro_dps = np.array([gx, gy, gz], dtype=float)
            return accel, gyro_dps, raw_buffer

def run_calibration(port=PORT, baudrate=115200, samples=50):
    print("=" * 60)
    print("IMU SENSOR CALIBRATION")
    print("Place the sensor COMPLETELY FLAT and STATIONARY on a table.")
    print("=" * 60)
    
    for i in range(3, 0, -1):
        print(f"Starting in {i} seconds...")
        time.sleep(1)
        
    print(f"\nCollecting {samples} samples...")
    
    accel_samples = []
    gyro_samples = []
    raw_buffer = bytearray()
    
    with serial.Serial(port, baudrate, timeout=1) as ser:
        ser.reset_input_buffer()
        time.sleep(0.1)
        
        while len(gyro_samples) < samples:
            accel, gyro_dps, raw_buffer = read_one_binary_packet(ser, raw_buffer)
            accel_samples.append(accel)
            gyro_samples.append(gyro_dps)
            
            if len(gyro_samples) % 50 == 0:
                print(f"Progress: {len(gyro_samples)} / {samples} samples")

    accel_data = np.array(accel_samples)
    gyro_data = np.array(gyro_samples)

    # Compute mean biases
    gyro_bias_dps = np.mean(gyro_data, axis=0)
    gyro_bias_rad = np.deg2rad(gyro_bias_dps)
    
    accel_mean = np.mean(accel_data, axis=0)
    accel_bias = accel_mean - np.array([0.0, 0.0, 1.0])
    
    return accel_bias, gyro_bias_rad

if (CALIBRATION):
  accel_bias, gyro_bias_rad = run_calibration(port=PORT, baudrate=115200, samples=CAL_SAMPLE_NUM)
  np.savez("imu_calibration.npz", gyro_bias_rad=gyro_bias_rad, accel_bias=accel_bias)
else:
  calib_data = np.load("imu_calibration.npz")
  gyro_bias_rad = calib_data["gyro_bias_rad"]
  accel_bias = calib_data["accel_bias"]

def binary_uart_reader(port=PORT, baudrate=115200):
    with serial.Serial(port, baudrate, timeout=0.1) as ser:
     
        ser.reset_input_buffer()
        raw_buffer = bytearray()
        
        while True:
            # Read available bytes in bulk
            waiting = ser.in_waiting
            if waiting > 0:
                raw_buffer.extend(ser.read(waiting))
            else:
                time.sleep(0.001)
                continue

            # Process all full packets in buffer, keep the newest
            latest_data = None
            while len(raw_buffer) >= PACKET_SIZE:
                # Find start header
                header_idx = raw_buffer.find(HEADER)
                
                if header_idx == -1:
                    # Header not found; clear everything except last byte
                    raw_buffer = raw_buffer[-1:]
                    break
                
                if header_idx > 0:
                    # Trim bytes before the header
                    del raw_buffer[:header_idx]

                if len(raw_buffer) < PACKET_SIZE:
                    break

                # Extract and unpack one frame
                packet_bytes = bytes(raw_buffer[:PACKET_SIZE])
                del raw_buffer[:PACKET_SIZE]
                
                _, _, ax, ay, az, gx, gy, gz, temp = struct.unpack(PACKET_FORMAT, packet_bytes)
                
                accel = np.array([ax, ay, az], dtype=float)
                accel -= accel_bias
                for i in range(3):
                  accel[i] = np.round(accel[i]*9.8, decimals=2)
                gyro_rad = np.deg2rad([gx, gy, gz])
                gyro_rad -= gyro_bias_rad
                gyro_deg = gyro_rad*(180.0/3.14)
                for i in range(3):
                  gyro_deg[i] = np.round(gyro_deg[i], decimals=2)
                latest_data = (accel, gyro_rad)
                latest_data_deg = (accel, gyro_deg)
                
            if latest_data is not None:
                if data_queue.full():
                    try:
                        data_queue.get_nowait()
                    except queue.Empty:
                        pass
                data_queue.put(latest_data)
                printOut = (f'accelerometer(m/s^2) x:{accel[0]}, y:{accel[1]}, z:{accel[2]} | gyro(deg/s) x:{gyro_deg[0]}, y:{gyro_deg[1]}, z:{gyro_deg[2]} | temp(C): {round(temp, 2)}')
                print(printOut)
               

scene.title = "High-Speed IMU 3D Tracker"
scene.width, scene.height = 800, 600
board = box(length=1, height=0.1, width=0.5, color=color.cyan)

madgwick = Madgwick(frequency=50.0)
q = np.array([1.0, 0.0, 0.0, 0.0])

def quaternion_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
    ])

threading.Thread(target=binary_uart_reader, args=(PORT, 115200), daemon=True).start()

while True:
    rate(120)
    if not data_queue.empty():
        accel, gyro_rad = data_queue.get()
        
        q = madgwick.updateIMU(q, gyr=gyro_rad, acc=accel)
        R = quaternion_to_R(q)
        
        board.axis = vector(R[0, 0], R[1, 0], R[2, 0])
        board.up   = vector(R[0, 1], R[1, 1], R[2, 1])