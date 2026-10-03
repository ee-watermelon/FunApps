# denoise_wav.py
import os
import torch
import torchaudio
import soundfile as sf
from torch import nn
from torch.utils.data import Dataset, DataLoader
from df.enhance import init_df, enhance, load_audio, save_audio

filePath = f'/noise_cancel/noisy_2.wav'
output_wav =  f'/noise_cancel/noisy_2_cleaned.wav'
cleanTrain = f'/noise_cancel/train/clean_train'
noisyTrain = f'/noise_cancel/train/noisy_train'
SAMPLE_RATE = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def preTrained_deepFilterNet(fileIn, fileOut):

  model, df_state, _ = init_df()
  audio, sr = load_audio(fileIn, sr=df_state.sr())
  enhanced_audio = enhance(model, df_state, audio)
  save_audio(fileOut, enhanced_audio, df_state.sr())

  print(f"Saved denoised audio to {fileOut}")


