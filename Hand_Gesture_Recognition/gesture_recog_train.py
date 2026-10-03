import os
import glob
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# =====================================================
# Directory Settings
# =====================================================
currentDIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = f"{currentDIR}/dataset/frames/"
MODEL_PATH = f"{currentDIR}/dataset/trained_model/gesture_lstm_model.pth"

# =====================================================
# CNN and LSTM Settings
# =====================================================
SEQ_LEN = 20
IMG_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 20
LR = 1e-4
CLASS_NAMES = ["dislike", "like", "stop"]
LSTM_FEATURE_SIZE = 512

# =====================================================
# CUDA Settings
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# =====================================================
# Transform
# =====================================================
def pad_to_square(img):
    w, h = img.size
    max_wh = max(w, h)

    pad_left = (max_wh - w) // 2
    pad_right = max_wh - w - pad_left
    pad_top = (max_wh - h) // 2
    pad_bottom = max_wh - h - pad_top

    return ImageOps.expand(
        img,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=(0, 0, 0)   # black padding
    )

image_transform = transforms.Compose([
    transforms.Lambda(pad_to_square),   # make square first
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# Dataset
# =====================================================
class GestureSequenceDataset(Dataset):
    def __init__(self, root_dir, class_names, seq_len=24, transform=None):
        self.root_dir = root_dir
        self.class_names = class_names
        self.seq_len = seq_len
        self.transform = transform
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
                    glob.glob(os.path.join(seq_dir, "*.jpeg"))
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
                0, len(frame_paths) - 1, self.seq_len
            ).long().tolist()
            selected_frames = [frame_paths[i] for i in indices]

        # If fewer than SEQ_LEN frames, repeat last frame
        else:
            selected_frames = frame_paths.copy()
            while len(selected_frames) < self.seq_len:
                selected_frames.append(frame_paths[-1])

        frames = []

        for path in selected_frames:
            image = Image.open(path).convert("RGB")

            if self.transform:
                image = self.transform(image)

            frames.append(image)

        frames = torch.stack(frames)  # shape: [seq_len, 3, H, W]

        return frames, torch.tensor(label, dtype=torch.long)


# =====================================================
# Model: ResNet18 feature extractor + LSTM
# =====================================================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=3, hidden_size=128, num_layers=1):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove final classification layer
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])

        self.feature_size = LSTM_FEATURE_SIZE

        self.lstm = nn.LSTM(
            input_size=self.feature_size,
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
        # x shape: [batch, seq_len, 3, H, W]
        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size * seq_len, c, h, w)

        features = self.cnn(x)
        features = features.view(batch_size, seq_len, self.feature_size)

        lstm_out, _ = self.lstm(features)

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
    seq_len=SEQ_LEN,
    transform=image_transform
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =====================================================
# Train setup
# =====================================================
model = CNN_LSTM(num_classes=len(CLASS_NAMES)).to(DEVICE)

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

    for frames, labels in train_loader:
        frames = frames.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = 100 * train_correct / train_total

    # Validation
    model.eval()

    val_correct = 0
    val_total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(frames)
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
    "seq_len": SEQ_LEN
}, MODEL_PATH)

print("Saved model:", MODEL_PATH)


# =====================================================
# Classify one gesture sequence folder
# =====================================================
def classify_sequence(sequence_folder):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    class_names = checkpoint["class_names"]
    seq_len = checkpoint["seq_len"]

    model = CNN_LSTM(num_classes=len(class_names)).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frame_paths = sorted(
        glob.glob(os.path.join(sequence_folder, "*.jpg")) +
        glob.glob(os.path.join(sequence_folder, "*.jpeg"))
    )

    if len(frame_paths) == 0:
        raise ValueError("No JPG/JPEG frames found.")

    if len(frame_paths) >= seq_len:
        indices = torch.linspace(0, len(frame_paths) - 1, seq_len).long().tolist()
        selected_frames = [frame_paths[i] for i in indices]
    else:
        selected_frames = frame_paths.copy()
        while len(selected_frames) < seq_len:
            selected_frames.append(frame_paths[-1])

    frames = []

    for path in selected_frames:
        image = Image.open(path).convert("RGB")
        image = image_transform(image)
        frames.append(image)

    frames = torch.stack(frames)
    frames = frames.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(frames)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    return class_names[predicted_idx.item()], confidence.item() * 100

