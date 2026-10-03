import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from collections import deque
import numpy as np
import mediapipe as mp
from torchvision import transforms, models


# =====================================================
# Directory Settings
# =====================================================
currentDIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{currentDIR}/dataset/trained_model/gesture_lstm_model.pth"

# =====================================================
# Mediapipe and LSTM Settings
# =====================================================
IMG_SIZE = 224
CONF_THRESHOLD = 60.0
LSTM_FEATURE_SIZE = 512

# =====================================================
# Camera Settings
# =====================================================
WINDOW_NAME = "LSTM Gesture Debug View"
CAMERA_ID = 0
CAMERA_DISPLAY_WIDTH = 1280 
CAMERA_DISPLAY_HEIGHT = 720
CAMERA_FRAME_RATE = 30      # frames/second

# =====================================================
# Color Settings
# =====================================================
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_RED    = (0, 0, 255)
COLOR_GREEN  = (0, 255, 0)
COLOR_BLUE   = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN   = (255, 255, 0)
COLOR_GRAY   = (80, 80, 80)

# =====================================================
# CUDA 
# =====================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print("Using device:", DEVICE)

# =====================================================
# Transform
# =====================================================
image_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# Model
# =====================================================
class CNN_LSTM(nn.Module):
    def __init__(self, num_classes=3, hidden_size=128, num_layers=1):
        super().__init__()

        resnet = models.resnet18(weights=None)
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
        batch_size, seq_len, c, h, w = x.shape

        x = x.view(batch_size * seq_len, c, h, w)

        features = self.cnn(x)
        features = features.view(batch_size, seq_len, self.feature_size)

        lstm_out, _ = self.lstm(features)
        last_output = lstm_out[:, -1, :]

        output = self.fc(last_output)

        return output


# =====================================================
# Overlay helper functions
# =====================================================
def draw_probability_bars(frame, class_names, probabilities, x=30, y=120):
    bar_width = 220
    bar_height = 22
    gap = 12

    for i, class_name in enumerate(class_names):
        prob = probabilities[i] * 100
        y_pos = y + i * (bar_height + gap)

        cv2.putText(
            frame,
            f"{class_name}: {prob:.1f}%",
            (x, y_pos - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_WHITE,
            1
        )

        cv2.rectangle(
            frame,
            (x, y_pos),
            (x + bar_width, y_pos + bar_height),
            COLOR_GRAY,
            1
        )

        filled_width = int(bar_width * probabilities[i])

        cv2.rectangle(
            frame,
            (x, y_pos),
            (x + filled_width, y_pos + bar_height),
            COLOR_GREEN,
            -1
        )


def draw_confidence_history(frame, history, x=30, y=280, w=300, h=100):
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GRAY, 1)

    cv2.putText(
        frame,
        "Confidence history",
        (x, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_WHITE,
        1
    )

    if len(history) < 2:
        return

    points = []

    for i, value in enumerate(history):
        px = x + int(i * w / max(1, len(history) - 1))
        py = y + h - int((value / 100.0) * h)
        points.append((px, py))

    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], COLOR_YELLOW, 2)

    # 60% threshold line
    threshold_y = y + h - int((CONF_THRESHOLD / 100.0) * h)
    cv2.line(frame, (x, threshold_y), (x + w, threshold_y), COLOR_RED, 1)


def draw_lstm_buffer_points(frame, buffer_len, seq_len, x=30, y=430):
    radius = 4
    gap = 10

    cv2.putText(
        frame,
        f"LSTM frame buffer: {buffer_len}/{seq_len}",
        (x, y - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        COLOR_WHITE,
        1
    )

    for i in range(seq_len):
        color = COLOR_GREEN if i < buffer_len else COLOR_GRAY
        cv2.circle(frame, (x + i * gap, y), radius, color, -1)


# =====================================================
# Load model
# =====================================================
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

class_names = checkpoint["class_names"]
seq_len = checkpoint["seq_len"]

model = CNN_LSTM(num_classes=len(class_names)).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Classes:", class_names)
print("Sequence length:", seq_len)


# =====================================================
# Live camera
# =====================================================
frame_buffer = deque(maxlen=seq_len)
confidence_history = deque(maxlen=80)

#cap = cv2.VideoCapture(CAMERA_ID)

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_DISPLAY_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_DISPLAY_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAMERA_FRAME_RATE)

print("Width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

print("Press q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (CAMERA_DISPLAY_WIDTH, CAMERA_DISPLAY_HEIGHT))
    if not ret:
        print("Failed to read frame.")
        break

    # Optional mirror effect for webcam
    #frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            for idx, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)

                cv2.circle(frame, (cx, cy), 4, COLOR_RED, -1)

                cv2.putText(
                    frame,
                    str(idx),
                    (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    COLOR_CYAN,
                    1
                )
    else:
        cv2.putText(
            frame,
            "No hand landmarks detected",
            (30, 520),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_RED,
            2
        )

    input_tensor = None

    if results.multi_hand_landmarks:
        h, w, _ = rgb_frame.shape

        hand_landmarks = results.multi_hand_landmarks[0]

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        x_min = int(min(xs) * w)
        x_max = int(max(xs) * w)
        y_min = int(min(ys) * h)
        y_max = int(max(ys) * h)

        # Add margin around hand
        margin = 60
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(w, x_max + margin)
        y_max = min(h, y_max + margin)

        hand_crop = rgb_frame[y_min:y_max, x_min:x_max]

        if hand_crop.size > 0:
            pil_image = Image.fromarray(hand_crop)
            input_tensor = image_transform(pil_image)

            frame_buffer.append(input_tensor)

            # Optional: show crop box
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                COLOR_BLUE,
                2
            )
    else:
        frame_buffer.clear()


    label_text = "Collecting frames..."
    confidence_text = ""
    probabilities_np = np.zeros(len(class_names), dtype=np.float32)

    if len(frame_buffer) == seq_len:
        sequence_tensor = torch.stack(list(frame_buffer))
        sequence_tensor = sequence_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1)[0]

        probabilities_np = probabilities.cpu().numpy()

        predicted_idx = int(np.argmax(probabilities_np))
        predicted_class = class_names[predicted_idx]
        confidence_value = probabilities_np[predicted_idx] * 100

        confidence_history.append(confidence_value)

        if confidence_value >= CONF_THRESHOLD:
            label_text = f"Gesture: {predicted_class}"
        else:
            label_text = ""

        confidence_text = f"Confidence: {confidence_value:.1f}%"

    # Main text
    cv2.putText(
        frame,
        label_text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        COLOR_GREEN,
        2
    )

    cv2.putText(
        frame,
        confidence_text,
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        COLOR_GREEN,
        2
    )

    # Debug overlays
    draw_probability_bars(frame, class_names, probabilities_np)
    draw_confidence_history(frame, confidence_history)
    draw_lstm_buffer_points(frame, len(frame_buffer), seq_len)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               