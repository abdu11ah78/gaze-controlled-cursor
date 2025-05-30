
# Gaze Controlled Virtual Mouse 🖱️👁️

This project implements a **Gaze-Controlled Virtual Mouse** that moves the cursor based on eye gaze direction using deep learning. The system uses a webcam to detect facial features and estimate gaze direction in real-time, allowing hands-free control of the mouse pointer.

## 🔍 Features

- Real-time webcam-based cursor control using gaze direction
- Blink detection for click simulation (optional)
- Supports three trained models:
  - **EfficientNet**
  - **GazeNet**
  - **Custom Gaze Model**
- Data augmentation for robust model performance
- Smoothing and fail-safe controls

---

## 🧠 Models

### 1. EfficientNet
- Transfer learning-based model trained on gaze data.
- Achieves high accuracy and generalization on test set.
  
### 2. GazeNet
- Custom CNN architecture designed for fast and efficient gaze prediction.

### 3. Custom Gaze Model
- Simpler architecture for experimentation and debugging.

---
## 🔗 Model Downloads



- [Download GazeNet model](https://drive.google.com/file/d/1EYciPXK06-M6cmTnxIsEgN7CxSdsZs05/view?usp=sharing)
- [Download EfficientNet model](https://drive.google.com/file/d/1Lull97iwqNHv0TsIqBs1Qo8ogzbfMzpz/view?usp=sharing)
- [Download Custom Gaze model](https://drive.google.com/file/d/1tPTdKGHxGaaZZuKFlG6fK2fxJn6LbMvV/view?usp=sharing)
## 🗃️ Dataset

Images were organized into classes representing different gaze directions. A data augmentation pipeline was applied to:
- Resize and normalize images to 128x128
- Apply random brightness, rotation, flips, zoom, shear, etc.

> All images were normalized and saved before training.

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- OpenCV
- TensorFlow (for augmentation)
- torchvision
- PIL
- scipy
- pyautogui

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## 📂 Repository Structure

```
.
├── models/
│   ├── efficientnet_model.pth
│   ├── gazenet_model.pth
│   └── custom_gaze_model.pth
├── gaze_control.py              # Main script for running gaze control
├── data_preprocessing.py       # Normalization and augmentation script
├── requirements.txt
├── README.md
└── sample_videos/              # (Optional) Demo recordings
```

---

## ▶️ How to Run

### 1. Data Preprocessing (Optional)

```bash
python data_preprocessing.py
```

This will normalize and augment your dataset.

### 2. Run Gaze-Controlled Mouse

```bash
python gaze_control.py
```

Press `Q` to quit the webcam window.

---

## 📊 Performance

| Model        | Accuracy | Notes                          |
| ------------ | -------- | ------------------------------ |
| EfficientNet | High     | Best generalization            |
| GazeNet      | Medium   | Fast and lightweight           |
| Custom Gaze  | Basic    | Good for debugging and testing |

---

## 📽️ Demo

Add demo video or GIF showing cursor movement based on gaze (upload in `sample_videos/` folder).

---

## 📌 Notes

* You may need to fine-tune the mapping between gaze predictions and screen resolution.
* Only tested on single-monitor setup.

---

## 📄 License

This project is open-source under the MIT License.
