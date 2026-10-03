
import sys
import os
import warnings
from scipy.signal.windows import get_window
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

currentDIR = os.path.dirname(os.path.abspath(__file__))
filePath = f'{currentDIR}/SineSweep1s.wav'

channel = 0             # 0 = left for stereo
mode = "welch"          # "single" or "welch" (for the frequency-response plot)
nfft =  16484        # 4096, 8192, 16484, 32768, 65536
window_name = "hann"    # "hann", "hamming", "blackman", "boxcar"
precision = "float64"   # "float32" to mimic C float
overlap = 0.8         # used only in welch mode
db_floor = -180.0       # clamp low-level floor in plot
show_phase = False      # optional: only supported in "single" mode below
FR_plot = 1

# ---- Tone metrics (SNR/THD/THD+N) ----
fundamental_freq = 1000   # Hz (known test tone frequency)
num_harmonics = 20        # how many harmonics to include in THD
guard_bins = 10     # include +/- this many bins around fundamental & each harmonic
band_limits = (20.0, 20000.0)  # measurement band for noise power; set to None for DC..Nyquist


def to_float_pcm(x):
   
    if np.issubdtype(x.dtype, np.floating):
        return x.astype(np.float64)
    if x.dtype == np.int16:
        return x / 32768.0
    if x.dtype == np.int32:
        return x / 2147483648.0
    if x.dtype == np.uint8:
        return (x.astype(np.int16) - 128) / 128.0
    maxv = np.max(np.abs(x))
    return x / maxv if maxv > 0 else x

def load_channel(wav_path, ch):
    fs, data = wavfile.read(wav_path)
    data = to_float_pcm(data)
    x = data[:, ch] if data.ndim == 2 else data
   
    peak = np.max(np.abs(x)) + 1e-20
    x = x / peak
    return fs, x

def rfft_mag_phase_db(x, fs, nfft, window_name, dtype, return_phase):
  
    win = get_window(window_name, x.shape[0], fftbins=True).astype(dtype)
    xw = (x * win).astype(dtype)
    X = np.fft.rfft(xw, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)

    cg = np.sum(win) / xw.size
    mag = np.abs(X) / (nfft * cg + np.finfo(dtype).eps)
    mag_db = 20.0 * np.log10(mag + np.finfo(dtype).eps)

    phase = np.unwrap(np.angle(X)) if return_phase else None
    return freqs, mag_db, phase

def welch_mag_db(x, fs, nfft, window_name, dtype, overlap):
 
    win = get_window(window_name, nfft, fftbins=True).astype(dtype)
    hop = int(nfft * (1 - overlap))
    hop = max(hop, 1)
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    segments = [x[i:i+nfft] for i in range(0, len(x) - nfft + 1, hop)]

    cg = np.sum(win) / nfft
    psd_accum = np.zeros(nfft//2 + 1, dtype=dtype)
    for seg in segments:
        segw = seg * win
        X = np.fft.rfft(segw, n=nfft)
        psd_accum += (np.abs(X) / (nfft * cg + np.finfo(dtype).eps))**2
    psd_mean = psd_accum / max(len(segments), 1)
    mag_db = 10.0 * np.log10(psd_mean + np.finfo(dtype).eps)
    freqs = np.fft.rfftfreq(nfft, d=1.0/fs)
    return freqs, mag_db

def compute_FR(test_audio):
 
  # ======== Compute & plot FR ========
  dtype = np.float32 if precision == "float32" else np.float64

  freqs_ref = None
  mag_list = []
  labels = []
  feats_FR = []
  comb = []
  freqs = []
  mag_db = []

  fs, x = load_channel(test_audio, channel)
  x = x.astype(dtype, copy=False)

  if mode == "single":
      if len(x) < nfft:
          x_use = np.pad(x, (0, nfft - len(x)))
      else:
          x_use = x[:nfft]
      freqs, mag_db, phase = rfft_mag_phase_db(x_use, fs, nfft, window_name, dtype, show_phase)
  else:
      freqs, mag_db = welch_mag_db(x, fs, nfft, window_name, dtype, overlap)
      phase = None

  if FR_plot == 1 :
    plt.figure()
  
    plt.semilogx(freqs, mag_db, linewidth=1)
    plt.ylim(-180, 0)
    plt.xlim(1, min(40000, int(freqs[-1])))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dBFS)")
    plt.title(f"Frequency Response ({mode.capitalize()}), NFFT={nfft}, window={window_name}")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    comb = np.array([freqs, mag_db])


  return comb

if __name__ == "__main__":
    compute_FR(filePath)