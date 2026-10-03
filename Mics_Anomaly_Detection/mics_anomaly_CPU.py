import time
import queue
import joblib
import numpy as np
import sounddevice as sd
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim


# =====================================================
# Settings
# =====================================================
SAMPLE_RATE = 16000
FRAME_SECONDS = 0.5
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_SECONDS)

MIC_DEVICES = [
    # {"name": "Device 1 mic", "device": 1},
    # {"name": "Device 2 mic", "device": 2},
    # {"name": "Device 4 mic", "device": 4},
]

TRAIN_SECONDS_PER_MIC = 30


# =====================================================
# Autoencoder
# =====================================================
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# =====================================================
# Feature extraction for ONE microphone
# =====================================================
def extract_features(audio):
    """
    audio shape: [samples, 1] or [samples]
    """

    x = np.squeeze(audio)

    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    zcr = np.mean(np.abs(np.diff(np.sign(x)))) / 2

    clipping_ratio = np.mean(np.abs(x) > 0.98)
    dropout_ratio = np.mean(np.abs(x) < 1e-5)

    freqs, psd = welch(x, fs=SAMPLE_RATE, nperseg=1024)

    low_energy = np.mean(psd[(freqs >= 100) & (freqs < 500)])
    mid_energy = np.mean(psd[(freqs >= 500) & (freqs < 3000)])
    high_energy = np.mean(psd[(freqs >= 3000) & (freqs < 7000)])

    return np.array([
        rms,
        peak,
        zcr,
        clipping_ratio,
        dropout_ratio,
        low_energy,
        mid_energy,
        high_energy
    ], dtype=np.float32)


# =====================================================
# File names
# =====================================================
def model_file(device_id):
    return f"mic_device_{device_id}_autoencoder.pt"


def scaler_file(device_id):
    return f"mic_device_{device_id}_scaler.pkl"


def threshold_file(device_id):
    return f"mic_device_{device_id}_threshold.npy"


# =====================================================
# Record one microphone
# =====================================================
def record_one_mic(device_id, seconds):
    print(f"\nRecording device {device_id} for {seconds} seconds...")

    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_id
    )
    with tqdm(total=seconds, desc="Recording", unit="s") as pbar:
      for _ in range(seconds):
        time.sleep(1)
        pbar.update(1)

    sd.wait()

    features = []
    total_frames = audio.shape[0] // FRAME_SAMPLES

    for k in range(total_frames):
        start = k * FRAME_SAMPLES
        end = start + FRAME_SAMPLES
        frame = audio[start:end]
        features.append(extract_features(frame))

    return np.array(features, dtype=np.float32)


# =====================================================
# Train one autoencoder per microphone
# =====================================================
def train_one_mic(mic):
    name = mic["name"]
    device_id = mic["device"]

    print("\n====================================================")
    print(f"Training {name}, Windows device ID = {device_id}")
    print("====================================================")
    print("Use normal working microphone condition during training.")

    X = record_one_mic(device_id, TRAIN_SECONDS_PER_MIC)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_size = X_scaled.shape[1]
    model = Autoencoder(input_size)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    for epoch in range(300):
        optimizer.zero_grad()

        y = model(X_tensor)
        loss = criterion(y, X_tensor)

        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss = {loss.item():.6f}")

    with torch.no_grad():
        y = model(X_tensor)
        errors = torch.mean((X_tensor - y) ** 2, dim=1).numpy()

    threshold = np.mean(errors) + 3 * np.std(errors)

    print(f"{name} threshold = {threshold:.6f}")

    torch.save(model.state_dict(), model_file(device_id))
    joblib.dump(scaler, scaler_file(device_id))
    np.save(threshold_file(device_id), threshold)


def train_all(MIC_DEVICES):
    for mic in MIC_DEVICES:
        train_one_mic(MIC_DEVICES[mic])

    print("\nAll three microphone models trained.")


# =====================================================
# Load all models
# =====================================================
def load_all_models():
    loaded = []

    for mic in MIC_DEVICES:
        device_id = MIC_DEVICES[mic]["device"]

        scaler = joblib.load(scaler_file(device_id))
        threshold = float(np.load(threshold_file(device_id)))

        input_size = scaler.mean_.shape[0]
        model = Autoencoder(input_size)
        model.load_state_dict(torch.load(model_file(device_id)))
        model.eval()

        loaded.append({
            "name": MIC_DEVICES[mic]["name"],
            "device": device_id,
            "model": model,
            "scaler": scaler,
            "threshold": threshold
        })

    return loaded


# =====================================================
# Test one microphone frame
# =====================================================
def test_one_mic(mic, MIC_DEVICES):
    device_id = mic["device"]
    model = mic["model"]
    scaler = mic["scaler"]
    threshold = mic["threshold"]

    audio = sd.rec(
        FRAME_SAMPLES,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device_id
    )

    sd.wait()

    features = extract_features(audio)
    features_scaled = scaler.transform([features])

    x = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        y = model(x).numpy()[0]

    error = np.mean((features_scaled[0] - y) ** 2)

    return error, features


# =====================================================
# Live test all microphones
# =====================================================
def test_all_live(MIC_DEVICES):
    models = load_all_models()

    print("\nStarting live test for devices 1, 2, and 4.")
    print("Press Ctrl+C to stop.\n")

    while True:
        print("----------------------------------------------------")

        worst_name = None
        worst_device = None
        worst_error_ratio = -1

        for mic in models:
            error, features = test_one_mic(mic, MIC_DEVICES)

            threshold = mic["threshold"]
            ratio = error / threshold if threshold > 0 else 0

            status = "ANOMALY" if error > threshold else "OK"

            print(
                f"{mic['name']} "
                f"(Windows device {mic['device']}): "
                f"error={error:.6f}, "
                f"threshold={threshold:.6f}, "
                f"ratio={ratio:.2f}, "
                f"status={status}"
            )

            if ratio > worst_error_ratio:
                worst_error_ratio = ratio
                worst_name = mic["name"]
                worst_device = mic["device"]

        if worst_error_ratio > 1.0:
            print(f">>> Most suspicious: {worst_name}, Windows device {worst_device}")
        else:
            print("System status: normal")

        time.sleep(0.5)


# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    mic_devices = []

    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            micName = dev["name"]
            mic_devices.append({
                "device": i,
                "name": dev["name"]
               # "channels": dev["max_input_channels"]
            })
            print(f"[mic number: {i}], mic name: {micName}")

    devList = input(f"\nType device number: ").strip().lower()   
    devList_arr = [int(d.strip()) for d in devList.split(",")]

    MIC_DEVICES = {k: mic_devices[k] for k in devList_arr}

    mode = input("\nType 'train' or 'test': ").strip().lower()

    if mode == "train":
        train_all(MIC_DEVICES)

    elif mode == "test":
        test_all_live(MIC_DEVICES)

    else:
        print("Invalid mode. Type 'train' or 'test'.")