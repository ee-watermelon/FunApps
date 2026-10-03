import os
#os.environ["OPENBLAS_NUM_THREADS"] = "24"
#os.environ["OMP_NUM_THREADS"] = "24"
#os.environ["MKL_NUM_THREADS"] = "24"
from threadpoolctl import threadpool_limits
import sys
import time
import threading 
import torch
import cupy as cp
from datetime import datetime
import csv
import matplotlib.pyplot as plt
from threadpoolctl import threadpool_info
import numpy as np
from pynvml import *

#####################################################################
SAVE_RESULT = 1       # 1:yes, 0:no
CSV_FILE = "gpu_bench_tab3.csv"
PLOT_RESULT = 0
running = True
power_log = []
#####################################################################

def power_monitor():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    while running:
        t = time.time()
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0
        power_log.append((t, power))
        #print(f"Power: {power:.2f} W")
        time.sleep(0.1)

    nvmlShutdown()

def integer_test(target, arrSize):
  match target:
    case 'CPU1':
      x = np.random.randint(0, 1000, size=arrSize, dtype = np.int32)
      
      with threadpool_limits(limits=1):
        start = time.perf_counter()
        y = np.sin(x) + np.cos(x)
        end = time.perf_counter()
      print ('CPU_1_integer_time: ', end-start, 'seconds')
      return (end-start), y
    
    case 'CPU24':
      x = np.random.randint(0, 1000, size=arrSize, dtype = np.int32)
      
      with threadpool_limits(limits=24):
        start = time.perf_counter()
        y = np.sin(x) + np.cos(x)
        end = time.perf_counter()
      print ('CPU_24_integer_time: ', end-start, 'seconds')
      return (end-start), y

    case 'GPUw':
      x = cp.random.randint(0, 1000, size=arrSize, dtype = cp.int32)
      cp.sin(x) + cp.cos(x)     #warmup
      
      cp.cuda.Stream.null.synchronize()
      start = time.perf_counter()
      y = cp.sin(x) + cp.cos(x)
      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      return (end-start), y

    case 'GPUnw':
      x = cp.random.randint(0, 1000, size=arrSize, dtype = cp.int32)
      
      cp.cuda.Stream.null.synchronize()
      start = time.perf_counter()
      y = cp.sin(x) + cp.cos(x)
      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      return (end-start), y

def float_test(target, arrSize):
  match target:
    case 'CPU1':
      x = np.random.random(arrSize).astype(np.float32)
      
      with threadpool_limits(limits=1):
        start = time.perf_counter()
        y = np.sin(x) + np.cos(x)
        end = time.perf_counter()
      print ('CPU_1_float_time: ', end-start, 'seconds')
      return (end-start), y
    
    case 'CPU24':
      x = np.random.random(arrSize).astype(np.float32)
      
      with threadpool_limits(limits=24):
        start = time.perf_counter()
        y = np.sin(x) + np.cos(x)
        end = time.perf_counter()
      print ('CPU_24_float_time: ', end-start, 'seconds')
      return (end-start), y
    
    case 'GPUw':
      x = cp.random.random(arrSize).astype(cp.float32)
      cp.sin(x) + cp.cos(x)     #warmup
      
      cp.cuda.Stream.null.synchronize()
      start = time.perf_counter()
      y = cp.sin(x) + cp.cos(x)
      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      print ('GPU_float_warmup_time: ', end-start, 'seconds')
      return (end-start), y

    case 'GPUnw':
      x = cp.random.random(arrSize).astype(cp.float32)
      
      cp.cuda.Stream.null.synchronize()
      start = time.perf_counter()
      y = cp.sin(x) + cp.cos(x)
      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      print ('GPU_float_no_warmup_time: ', end-start, 'seconds')
      return (end-start), y

def matrix_multiplication(target, rowSize, colSize):
  match target:
    case 'CPU1':
      x = np.random.rand(rowSize, colSize).astype(np.float32)
      y = np.random.rand(rowSize, colSize).astype(np.float32)
      
      with threadpool_limits(limits=1):
        start = time.perf_counter()
        z = np.matmul(x, y)
        end = time.perf_counter()
      print ('CPU_1_matrix_float_time: ', end-start, 'seconds')
      return (end-start), z
    
    case 'CPU24':
      x = np.random.rand(rowSize, colSize).astype(np.float32)
      y = np.random.rand(rowSize, colSize).astype(np.float32)
      
      with threadpool_limits(limits=24):
        start = time.perf_counter()
        z = np.matmul(x, y)
        end = time.perf_counter()
      print ('CPU_24_matrix_float_time: ', end-start, 'seconds')
      return (end-start), z

    case 'GPUw':
      x = cp.random.rand(rowSize, colSize, dtype=cp.float32)
      y = cp.random.rand(rowSize, colSize, dtype=cp.float32)
      cp.matmul(x, y)     #warmup
      
      cp.cuda.Stream.null.synchronize()
      start = time.perf_counter()
      z = cp.matmul(x, y)
      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      print ('GPU_matrix_float_warmup_time: ', end-start, 'seconds')
      return (end-start), z

    case 'GPUnw':
      x = cp.random.rand(rowSize, colSize, dtype=cp.float32)
      y = cp.random.rand(rowSize, colSize, dtype=cp.float32)
      
      cp.cuda.Stream.null.synchronize()
      start = time.perf_counter()
      z = cp.matmul(x, y)
      cp.cuda.Stream.null.synchronize()
      end = time.perf_counter()
      print ('GPU_matrix_float_no_warmup_time: ', end-start, 'seconds')
      return (end-start), z
   
def performance_test(target, testType, arrSize):
  arr_out = []
  match testType:
    case 'integer':
      tTime, arr_out = integer_test(target, arrSize)
      return tTime, arr_out
    case 'float':
      tTime, arr_out = float_test(target, arrSize)
      return tTime, arr_out

def GPU_mem_bandwidth_test(arraySize) -> float:
  device = 'cuda'
  x = torch.randn(500_000_000, device = device, dtype = torch.float32)
  
  torch.cuda.synchronize()
  start = time.perf_counter()
  y = x * 2
  torch.cuda.synchronize()
  end = time.perf_counter()
  bandwidth = x.numel() * 4 * 2 / (end-start) / 1e9
  return bandwidth

def save_to_csv(csv_file_path, task, target, dataType, arraySize, testTime, bandWidth):
  write_header = not os.path.exists(csv_file_path)
  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M") 
  with open(csv_file_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
      writer.writerow(["timestamp", "array size", "target", "condition", "task", "data type", "execution time(s)"])
      write_header = False
    writer.writerow([timestamp, task, target, dataType, arraySize, testTime, bandWidth])
  return

def matrix_multiplication_benchmark():
  cpuArr1 = [] 
  cpuArr24 = [] 
  gpuArr_w = []
  gpuArr_nw = []
  arr_out = []
  arrSize = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 40000]

  for i in arrSize:
    tTime, arr_out = matrix_multiplication('GPUnw', i, i)
    gpuArr_nw.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE,  i, "GPU", "no warmup", "matrix mult", "float", tTime)

  for i in arrSize:
    tTime, arr_out = matrix_multiplication('GPUw', i, i)
    gpuArr_w.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "GPU", "warmup", "matrix mult", "float", tTime)

  for i in arrSize:
    tTime, arr_out = matrix_multiplication('CPU1', i, i)
    cpuArr1.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "CPU", "1 thread", "matrix mult", "float", tTime)
    
  for i in arrSize:
    tTime, arr_out = matrix_multiplication('CPU24', i, i)
    cpuArr24.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "CPU", "24 thread", "matrix mult", "float", tTime)
  
  if (PLOT_RESULT):
    plt.plot(arrSize, cpuArr1, label = 'CPU_1 thread')
    plt.plot(arrSize, cpuArr24, label = 'CPU_24 threads')
    plt.plot(arrSize, gpuArr_w, label = 'GPU_warmup')
    plt.plot(arrSize, gpuArr_nw, label = 'GPU_no_warmup')
    plt.title('[Windows] matrix multiplication benchmark')
    plt.ylabel('Time(seconds)')
    plt.xlabel('Matrix Size')
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(visible=True)
    plt.legend()
    plt.tight_layout()
    plt.show()

  return arrSize, gpuArr_nw, gpuArr_w, cpuArr1, cpuArr24

def performance_benchmark():
  cpuIntArr1 = [] 
  cpuFloatArr1 = []
  cpuIntArr24 = [] 
  cpuFloatArr24 = []
  gpuIntArr_w = []
  gpuFloatArr_w = []
  gpuIntArr_nw = []
  gpuFloatArr_nw = []
  arr_out = []
  arrSize = [10, 50, 100, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 40000]

  for i in arrSize:
    tTime, arr_out = performance_test('GPUnw', 'integer', i)
    gpuIntArr_nw.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "GPU", "no warmup", "performance", "integer", tTime)

    tTime, arr_out = performance_test('GPUnw', 'float', i)
    gpuFloatArr_nw.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, " GPU", "no warmup", "performance", "float", tTime)

    tTime, arr_out = performance_test('GPUw', 'integer', i)
    gpuIntArr_w.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "GPU", "warmup", "performance", "integer", tTime)

    tTime, arr_out = performance_test('GPUw', 'float', i)
    gpuFloatArr_w.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "GPU", "warmup", "performance", "float", tTime)

    tTime, arr_out = performance_test('CPU1', 'integer', i)
    cpuIntArr1.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "CPU", "1 thread", "performance", "integer", tTime)
    
    tTime, arr_out = performance_test('CPU1', 'float', i)
    cpuFloatArr1.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "CPU", "1 thread", "performance", "float", tTime)
    
    tTime, arr_out = performance_test('CPU24', 'integer', i)
    cpuIntArr24.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "CPU", "24 thread", "performance", "integer", tTime)

    tTime, arr_out = performance_test('CPU24', 'float', i)
    cpuFloatArr24.append(tTime)
    if (SAVE_RESULT):
      save_to_csv(CSV_FILE, i, "CPU", "24 thread", "performance", "float", tTime)

  scaled_arrSize = np.array(arrSize) / 1000

  if (PLOT_RESULT):
    plt.plot(scaled_arrSize, cpuIntArr1, label = 'CPU_integer_1 thread')
    plt.plot(scaled_arrSize, cpuFloatArr1, label = 'CPU_float_1 thread')
    plt.plot(scaled_arrSize, cpuIntArr24, label = 'CPU_integer_24 threads')
    plt.plot(scaled_arrSize, cpuFloatArr24, label = 'CPU_float_24 threads')
    plt.plot(scaled_arrSize, gpuIntArr_w, label = 'GPU_integer_warmup')
    plt.plot(scaled_arrSize, gpuFloatArr_w, label = 'GPU_float_warmup')
    plt.plot(scaled_arrSize, gpuIntArr_nw, label = 'GPU_integer_no_warmup')
    plt.plot(scaled_arrSize, gpuFloatArr_nw, label = 'GPU_float_no_warmup')
    plt.grid(visible = True)
    plt.title('[Windows] CPU & GPU Performance Tests (Integer, Float)')
    plt.ylabel('Time(seconds)')
    plt.xlabel('Input Array Size(x1e3)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.legend()
    plt.show()


  return scaled_arrSize, cpuFloatArr1, cpuFloatArr24, gpuFloatArr_w, gpuFloatArr_nw

def GPU_mem_bandwidth_benchmark():
  bwArr = []
  arrSize = [1000, 10000, 100_000, 500_000, 1000_000, 5000_000, 10_000_000, 50_000_000, 100_000_000, 500_000_000] 

  for i in arrSize: 
    bw = GPU_mem_bandwidth_test(arraySize=i)
    bwArr.append(bw)
    
  
  scaled_arrSize = np.array(arrSize) / 1000_000
  
  if (PLOT_RESULT):
    plt.plot(scaled_arrSize, bwArr)
    plt.title('[Windows] GPU Memory Bandwidth Benchmark (Float)')
    plt.ylabel('Bandwidth(GB/s)')
    plt.xlabel('Input Array Size(x1e6)')
    plt.ticklabel_format(style='plain', axis='x')
    plt.grid(visible=True)
    plt.show()

with open("GPU_W.csv", "a", newline="") as f:
  writer = csv.writer(f)
  for i in range(50):
    tTime, arr_out = performance_test('GPUw', 'float', 500)
    writer.writerow(arr_out)

# with open("GPU_N_W.csv", "a", newline="") as f:
#   writer = csv.writer(f)
#   for i in range(50):
#     tTime, arr_out = performance_test('GPUnw', 'float', 500)
#     writer.writerow(arr_out)  

performance_benchmark()
matrix_multiplication_benchmark()