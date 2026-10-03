import numpy as np
import ctypes
import matplotlib.pyplot as plt
import os

currentDIR = os.path.dirname(os.path.abspath(__file__))
dll_path = f'{currentDIR}/fir_dll/fir_dll.dll'

fs = 100
duration = 5
num_taps = 32
x_size = fs * duration
t_pre = np.arange(x_size) / fs
preData = (
        np.sin(2 * np.pi * 1 * t_pre)
        + 0.3 * np.sin(2 * np.pi * 8 * t_pre)
    ).astype(np.float32)
postData = np.zeros(x_size+num_taps-1, dtype=np.float32)
coeffTaps = np.ones(num_taps, dtype=np.float32) / num_taps
t_post = np.arange(x_size + num_taps - 1) / fs
def import_DLL(dll_path,preData, tapArr, postData, x_size, num_taps):   
   
    dll = ctypes.CDLL(dll_path)

    dll.FIR.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
        ctypes.c_int,
        ctypes.c_int
    ]

    dll.FIR.restype = None

    dll.FIR(preData, coeffTaps, postData, x_size, num_taps)

    return postData
    return postData

#     plt.plot(t_pre, preData, label="pre-FIR")
#     plt.plot(t_post, postData, label="post-FIR")
#     plt.grid(True)
#     plt.title("[Windows] FIR")
#     plt.xlabel("Time (s)")
#     plt.ylabel("Amplitude")
#     plt.legend()
#     plt.show()

# import_DLL(preData, coeffTaps, postData, x_size, num_taps)
