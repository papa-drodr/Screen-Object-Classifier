# Screen Object Classifier

A real-time screen capture-based object classifier. It captures a specific region of the screen, classifies it using a PyTorch model, and overlays the prediction results with visual feedback.

---

## Features

- **Real-time Screen Capture**: Captures a designated screen region (1080×720) in real time using `mss`
- **Classification Inference**: Classifies each captured frame through a pretrained `ScreenClassifier` model
- **Temporal Smoothing**: Averages softmax probabilities over the last 10 frames to stabilize predictions
- **Top-K Visualization**: Displays the top 3 predicted classes with probability bar charts overlaid on the screen
- **Minimap**: Shows a scaled-down view of the full screen with a red bounding box indicating the current capture region
- **Movable Capture Area**: Move the capture region in real time using WASD keys

---

## Requirements

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- mss
- NumPy

```bash
pip install torch opencv-python mss numpy
```

---

## Project Structure

```
.
├── main.py          # Capture loop and visualization logic
├── model.py         # ScreenClassifier model definition
└── README.md
```

---

## Usage

```bash
python main.py
```

---

## Controls

| Key | Action |
|---|---|
| `W` / `A` / `S` / `D` | Move capture area up / left / down / right |
| `N` | Refresh minimap background |
| `Q` / `ESC` | Quit |

---
## How It Works

1. On startup, the program captures the entire screen once to generate the minimap background.
2. Each frame, it captures the designated region (starting at coordinates `(150, 150)` by default).
3. The captured frame is fed into `ScreenClassifier`, and the softmax probabilities are averaged over the last 10 frames (smoothing).
4. The top 3 predicted classes are displayed with bar charts on the overlay.
5. The minimap highlights the current capture region with a red bounding box.