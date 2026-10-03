import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mediapipe as mp
import cv2

from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


currentDIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# Settings
# =====================================================
DATA_DIR = f"{currentDIR}/dataset/frames/"
MODEL_PATH = f"{currentDIR}/dataset/trained_model/gesture_lstm_model.pth"

SEQ_LEN = 20
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4

CLASS_NAMES = ["dislike", "like", "stop"]

NUM_LANDMARKS = 21
LANDMARK_DIMS = 3
INPUT_SIZE = NUM_LANDMARKS * LANDMARK_DIMS   # 21 x 3 = 63

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# =====================================================
# MediaPipe setup
# =====================================================
from mediapipe.tasks.python import vision
from mediapipe.tasks import python

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5
)
# =====================================================
# Landmark extraction
# =====================================================
def extract_hand_landmarks(image_path):
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Failed to load image: {image_path}")
        return np.zeros(INPUT_SIZE, dtype=np.float32)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = np.ascontiguousarray(image_rgb)

    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return np.zeros(INPUT_SIZE, dtype=np.float32)

    hand_landmarks = results.multi_hand_landmarks[0]

    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    return np.array(landmarks, dtype=np.float32)


# =====================================================
# Dataset
# =====================================================
class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, class_names, seq_len=20):
        self.hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
            )
        self.root_dir = root_dir
        self.class_names = class_names
        self.seq_len = seq_len
        self.samples = []

        for label_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, class_name)

            sequence_dirs = sorted([
                d for d in glob.glob(os.path.join(class_dir, "*"))
                if os.path.isdir(d)
            ])

            for seq_dir in sequence_dirs:
                frame_paths = sorted(
                    glob.glob(os.path.join(seq_dir, "*.jpg")) +
                    glob.glob(os.path.join(seq_dir, "*.jpeg")) +
                    glob.glob(os.path.join(seq_dir, "*.png"))
                )

                if len(frame_paths) > 0:
                    self.samples.append((frame_paths, label_idx))

        print("Total sequences:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]

        # If more than SEQ_LEN frames, sample evenly
        if len(frame_paths) >= self.seq_len:
            indices = torch.linspace(
                0,
                len(frame_paths) - 1,
                self.seq_len
            ).long().tolist()

            selected_frames = [frame_paths[i] for i in indices]

        # If fewer than SEQ_LEN frames, repeat last frame
        else:
            selected_frames = frame_paths.copy()

            while len(selected_frames) < self.seq_len:
                selected_frames.append(frame_paths[-1])

        sequence_landmarks = []

        for path in selected_frames:
            landmarks = extract_hand_landmarks(path)
            sequence_landmarks.append(landmarks)

        sequence_landmarks = np.stack(sequence_landmarks)

        # shape: [seq_len, 63]
        sequence_tensor = torch.tensor(sequence_landmarks, dtype=torch.float32)

        return sequence_tensor, torch.tensor(label, dtype=torch.long)


# =====================================================
# Model: Landmark LSTM
# =====================================================
class Landmark_LSTM(nn.Module):
    def __init__(
        self,
        input_size=INPUT_SIZE,
        num_classes=3,
        hidden_size=128,
        num_layers=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [batch, seq_len, 63]

        lstm_out, _ = self.lstm(x)

        # Use last frame output
        last_output = lstm_out[:, -1, :]

        output = self.fc(last_output)

        return output


# =====================================================
# Load data
# =====================================================
dataset = GestureSequenceDataset(
    root_dir=DATA_DIR,
    class_names=CLASS_NAMES,
    seq_len=SEQ_LEN
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================================================
# Train setup
# =====================================================
model = Landmark_LSTM(num_classes=len(CLASS_NAMES)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# =====================================================
# Training loop
# =====================================================
for epoch in range(EPOCHS):
    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for sequences, labels in train_loader:
        sequences = sequences.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(sequences)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # =================================================
    # Validation
    # =================================================
    model.eval()

    val_correct = 0
    val_total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(sequences)
            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    val_acc = 100 * val_correct / val_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Val Acc: {val_acc:.2f}%"
    )


# =====================================================
# Final Confusion Matrix
# =====================================================
print("\nFinal Validation Confusion Matrix")
print("Rows = Actual class")
print("Columns = Predicted class")
print(CLASS_NAMES)

cm = confusion_matrix(
    all_labels,
    all_preds,
    labels=list(range(len(CLASS_NAMES)))
)

print(cm)

print("\nClassification Report")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0
    )
)


# =====================================================
# Save model
# =====================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": CLASS_NAMES,
    "seq_len": SEQ_LEN,
    "input_size": INPUT_SIZE,
    "num_landmarks": NUM_LANDMARKS,
    "landmark_dims": LANDMARK_DIMS
}, MODEL_PATH)

print("Saved model:", MODEL_PATH)


# =====================================================
# Classify one gesture sequence folder
# =====================================================
def classify_sequence(sequence_folder):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    seq_len = checkpoint["seq_len"]
    input_size = checkpoint["input_size"]

    model = Landmark_LSTM(
        input_size=input_size,
        num_classes=len(class_names)
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frame_paths = sorted(
        glob.glob(os.path.join(sequence_folder, "*.jpg")) +
        glob.glob(os.path.join(sequence_folder, "*.jpeg")) +
        glob.glob(os.path.join(sequence_folder, "*.png"))
    )

    if len(frame_paths) == 0:
        raise ValueError("No image frames found.")

    if len(frame_paths) >= seq_len:
        indices = torch.linspace(
            0,
            len(frame_paths) - 1,
            seq_len
        ).long().tolist()

        selected_frames = [frame_paths[i] for i in indices]

    else:
        selected_frames = frame_paths.copy()

        while len(selected_frames) < seq_len:
            selected_frames.append(frame_paths[-1])

    sequence_landmarks = []

    for path in selected_frames:
        landmarks = extract_hand_landmarks(path)
        sequence_landmarks.append(landmarks)

    sequence_landmarks = np.stack(sequence_landmarks)

    sequence_tensor = torch.tensor(
        sequence_landmarks,
        dtype=torch.float32
    )

    # shape: [1, seq_len, 63]
    sequence_tensor = sequence_tensor.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(sequence_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return class_names[predicted_idx.item()], confidence.item() * 100