# Real-Time Webcam Fire Detection System

A lightweight, modular, and easy-to-understand real-time fire detection system built with **Python**, **OpenCV**, and **Ultralytics YOLO**.

This project detects fire and flames from a laptop webcam or external camera, calculates bounding boxes, confidence, real-time FPS, directional guidance (`LEFT` / `CENTER` / `RIGHT`), apparent fire size (area), and includes temporal multi-frame confirmation to eliminate false alarms. It is structured for straightforward future integration into a fire-fighting robot's Deep Q-Network (DQN) navigation system.

---

## 1. Project Structure

```text
fire-detection/
│
├── main.py              # Main webcam capture loop, HUD rendering & robot telemetry
├── detector.py          # FireDetector class: YOLO inference, tracking & filtering
├── requirements.txt     # Python dependency list
├── models/
│   ├── best.pt          # YOLOv8n fine-tuned fire/smoke model weights (~6.2 MB)
│   └── fire_weights.pt  # Fine-tuned fire & smoke model weights (~87.6 MB)
└── README.md            # Documentation and student guide
```

---

## 2. Models Supported

The detector seamlessly supports both weights:
- **`models/fire_weights.pt`** (Primary default): Classes `{0: 'fire', 1: 'smoke'}`
- **`models/best.pt`** (Lightweight YOLOv8n fallback): Classes `{0: 'smoke', 1: 'fire'}`

Target classes automatically map `fire`, `flame`, and `open_flame` aliases so detection is robust across models.

---

## 3. Installation & Environment

### Step 1: Open Terminal / PowerShell in the Project Directory
```bash
cd c:\Users\naman\Downloads\Yolo_model
```

### Step 2: Virtual Environment & GPU Support
If you have an NVIDIA GPU (e.g., RTX 3050):
```bash
# In your virtual environment:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install ultralytics opencv-python numpy
```
Or with standard pip:
```bash
pip install -r requirements.txt
```

---

## 4. How to Run

### Basic Run (Auto-detects GPU if available, else CPU):
```bash
python main.py
```
Press **`Q`** or **`ESC`** in the video window to quit cleanly.

---

## 5. Configuration Options

### A. Choose Inference Device (GPU vs CPU)
By default, `--device auto` uses your NVIDIA GPU (`device=0`) if CUDA is installed, falling back to CPU:
```bash
# Force NVIDIA GPU (device 0)
python main.py --device 0

# Force CPU inference
python main.py --device cpu
```

### B. Change Webcam Index
If your laptop has multiple cameras or you are using an external USB webcam:
```bash
# Try camera 1 (external webcam)
python main.py --camera 1

# Try camera 2
python main.py --camera 2
```

### C. Change Confidence Threshold
The confidence threshold filters out weak detections (default is `0.35`):
```bash
# Set threshold to 50%
python main.py --conf 0.50

# Set lower threshold (25%) for small flames / candle wick
python main.py --conf 0.25
```

### D. Change Multi-Frame Confirmation Requirement
Adjust how many consecutive positive frames are required before declaring `FIRE CONFIRMED`:
```bash
# Require 4 consecutive frames (default)
python main.py --frames 4
```

### E. Switch Between Models
```bash
# Use larger fire_weights model
python main.py --model models/fire_weights.pt

# Use lightweight YOLOv8n best.pt model
python main.py --model models/best.pt
```

### F. Optional: Detect Smoke Alongside Fire
```bash
python main.py --smoke
```

### G. Optional: Test with a Video File
```bash
python main.py --video path/to/sample_fire_video.mp4
```

---

## 6. How the System Works (Student Explanation)

### Pipeline Architecture

```text
Webcam
   ↓
OpenCV frame
   ↓
YOLO (GPU device=0 or CPU)
   ↓
Fire bounding box
   ↓
Confidence filtering (conf >= threshold)
   ↓
Multi-frame confirmation (streak >= 4 frames)
   ↓
Fire direction (LEFT / CENTER / RIGHT) + position (cx, cy) + size (area)
```

### 1. How YOLO Detects Fire
1. The camera captures continuous BGR video frames using OpenCV (`cv2.VideoCapture`).
2. Each frame is passed to the YOLO neural network running on GPU (`device=0`) or CPU.
3. The network predicts bounding boxes `[x1, y1, x2, y2]`, class identities, and confidence scores across the frame.
4. Internal Non-Maximum Suppression (NMS) removes duplicate overlapping boxes.

### 2. Multi-Frame Confirmation (Temporal Filtering)
- Single-frame detections can occasionally occur due to reflections or light glare.
- `detector.py` maintains an internal counter (`consecutive_fire_frames`):
  - If fire is seen in the current frame: `consecutive_fire_frames += 1`.
  - If no fire is seen: the counter immediately resets to `0`.
  - Only when `consecutive_fire_frames >= 4` does the status switch to:
    $$\text{FIRE CONFIRMED!}$$
- In the HUD, you can see `VERIFYING (k/4)` as the detector builds confidence.

### 3. LEFT / CENTER / RIGHT Direction Calculation
We calculate the horizontal center ($c_x$) of the strongest fire detection:

$$c_x = \frac{x_1 + x_2}{2}$$

The frame width $W$ is divided into three equal vertical bands:

```text
0                   W/3                 2W/3                  W
|--------------------|--------------------|-------------------|
|        LEFT        |       CENTER       |       RIGHT       |
|--------------------|--------------------|-------------------|
```

- If $c_x < \frac{W}{3} \implies \text{FIRE\_DIRECTION = LEFT}$ (Robot turns left)
- If $\frac{W}{3} \le c_x < \frac{2W}{3} \implies \text{FIRE\_DIRECTION = CENTER}$ (Robot moves straight)
- If $c_x \ge \frac{2W}{3} \implies \text{FIRE\_DIRECTION = RIGHT}$ (Robot turns right)

### 4. Apparent Fire Area Calculation
We calculate the area of the detected bounding box in pixels:

$$\text{width} = x_2 - x_1, \quad \text{height} = y_2 - y_1$$
$$\text{area} = \text{width} \times \text{height}$$

- When far away, the flame appears small.
- As the robot approaches, the flame occupies a larger pixel area, enabling optimal extinguishing standoff distance.

---

## 7. Robot DQN Navigation Integration

To connect this detection system to your Reinforcement Learning (DQN) agent, look in **`main.py`** at **Step 2**:

```python
# --- Step 2: Telemetry Data for Robot Navigation ---
# [HOOK FOR ROBOT DQN NAVIGATION SYSTEM]
is_fire_confirmed = result["confirmed"]         # True / False
direction_str     = result["fire_direction"]     # 'LEFT', 'CENTER', 'RIGHT', 'NONE'

if result["strongest"]:
    cx, cy    = result["strongest"]["center"]    # Center coordinate (x, y)
    fire_area = result["strongest"]["area"]      # Apparent area (size proxy for distance)
    fire_conf = result["strongest"]["confidence"]

    # DQN state vector:
    # state = [
    #     1.0 if is_fire_confirmed else 0.0,
    #     cx / frame.shape[1],                     # Normalized X [0.0 - 1.0]
    #     cy / frame.shape[0],                     # Normalized Y [0.0 - 1.0]
    #     fire_area / (frame.shape[0] * frame.shape[1]), # Normalized area
    # ]
    # action = dqn_agent.act(state)
    # robot_hardware.send_motor_command(action)
```
