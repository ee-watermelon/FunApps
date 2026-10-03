import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt


fs = 100
duration = 5.0
x_arr = np.linspace(0, 1, fs*int(duration))
t = np.arange(0, duration, 1 / fs)
num_taps = 16
h = np.ones(num_taps) / num_taps
x = np.sin(2*np.pi*1*t) + 0.3*np.sin(2*np.pi*8*t)

def fir_filter_gpu(x, h):
    """
    FIR filter on GPU.

    x: input signal
    h: FIR coefficients / taps

    y[n] = h[0]x[n] + h[1]x[n-1] + ... + h[M-1]x[n-M+1]
    """

    # Move data to GPU
    x_gpu = cp.asarray(x, dtype=cp.float32)
    h_gpu = cp.asarray(h, dtype=cp.float32)

    # FIR filtering using GPU convolution
    y_gpu = cp.convolve(x_gpu, h_gpu, mode="same")

    # Make sure GPU is finished before timing / returning
    cp.cuda.Stream.null.synchronize()

    # Move result back to CPU
    return cp.asnumpy(y_gpu)


# ============================================================
# Example usage
# ============================================================

def fir_gpu_python(x):

    # Simple moving-average FIR low-pass filter
    num_taps = 64
    h = np.ones(num_taps, dtype=np.float32) / num_taps

    start = time.time()
    y = fir_filter_gpu(x, h)
    end = time.time()

    tTime = end -start
    # plt.plot(x_arr, x, label = 'pre-FIR')
    # plt.plot(x_arr, y, label = 'post-FIR')
    # plt.grid(visible = True)
    # plt.title('[Windows] FIR')
    # plt.ylabel('Amplitude')
    # plt.xlabel('Time(s)')
    # plt.ticklabel_format(style='plain', axis='x')
    # plt.legend()
    # plt.show()
  
 

    return y, tTime

# fir_gpu_python(x)