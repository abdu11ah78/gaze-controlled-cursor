import pyautogui
import cv2
import numpy as np
from scipy.spatial import distance
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn

pyautogui.FAILSAFE = False  # Disable fail-safe (use with caution)

# ----------------------------- Model Definition -----------------------------
class GazeNet(nn.Module):
    def __init__(self, output_dim=236):  # Predict (x, y) gaze coordinates
        super(GazeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----------------------------- Load Model -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GazeNet(output_dim=236).to(device)
model.load_state_dict(torch.load("C:/Users/abdul/gaze_model_gazenet.pth", map_location=device))
model.eval()

# ----------------------------- Preprocessing -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ----------------------------- Blink Detection -----------------------------
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES = 3
blink_counter = 0
click_triggered = False

# ----------------------------- Webcam & Face Detection -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize previous gaze position (for smoothing)
previous_x, previous_y = screen_width // 2, screen_height // 2

# ----------------------------- Main Loop -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from webcam")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Approximate eye regions
        left_eye_region = frame[y + int(h * 0.4):y + int(h * 0.6), x + int(w * 0.1):x + int(w * 0.4)]
        right_eye_region = frame[y + int(h * 0.4):y + int(h * 0.6), x + int(w * 0.6):x + int(w * 0.9)]

        # Preprocess eyes
        left_eye_image = Image.fromarray(cv2.resize(left_eye_region, (224, 224)))
        right_eye_image = Image.fromarray(cv2.resize(right_eye_region, (224, 224)))

        left_eye_tensor = transform(left_eye_image).unsqueeze(0).to(device)
        right_eye_tensor = transform(right_eye_image).unsqueeze(0).to(device)

        with torch.no_grad():
            left_gaze = model(left_eye_tensor).cpu().numpy()[0]
            right_gaze = model(right_eye_tensor).cpu().numpy()[0]

        # Average gaze predictions
        gaze_x = (left_gaze[0] + right_gaze[0]) / 2.0
        gaze_y = (left_gaze[1] + right_gaze[1]) / 2.0

        # Map gaze coordinates to screen
        screen_x = int(gaze_x * screen_width)
        screen_y = int(gaze_y * screen_height)

        # Smoothing
        screen_x = int(previous_x * 0.7 + screen_x * 0.3)
        screen_y = int(previous_y * 0.7 + screen_y * 0.3)

        # Clamp to screen bounds
        screen_x = max(0, min(screen_x, screen_width - 1))
        screen_y = max(0, min(screen_y, screen_height - 1))

        pyautogui.moveTo(screen_x, screen_y)
        previous_x, previous_y = screen_x, screen_y

        # Draw rectangles on eyes
        cv2.rectangle(frame, (x + int(w * 0.1), y + int(h * 0.4)), (x + int(w * 0.4), y + int(h * 0.6)), (255, 0, 0), 2)
        cv2.rectangle(frame, (x + int(w * 0.6), y + int(h * 0.4)), (x + int(w * 0.9), y + int(h * 0.6)), (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Gaze Controlled Cursor", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
