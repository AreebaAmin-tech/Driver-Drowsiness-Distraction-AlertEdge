<div align="center">

# 🚗 Edge Driver Drowsiness & Distraction Alert System

**A lightweight, real-time Computer Vision safety pipeline engineered for edge devices.**  
Monitors driver vigilance, micro-sleeps, yawning, head gaze orientation, and mobile phone usage with sub-second latency and zero frame lag.

---

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-FaceMesh-00A67E?style=for-the-badge&logo=google&logoColor=white)](https://developers.google.com/mediapipe)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ONNX%20Runtime-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📌 Overview

Driver fatigue and distraction are leading contributors to road collisions worldwide. This system provides a **non-intrusive, vision-based Driver Monitoring System (DMS)** that evaluates driver cognitive alertness continuously in real-time.

By combining facial geometric landmarks with an optimized edge object detector, the system detects acute events (such as sudden eye closures or phone handling) while tracking cumulative, progressive drowsiness over a rolling 60-second temporal window.

---

## ⚡ Core Features

* **⚡ Ultra-Fast Micro-Sleep Detection**: Eye Aspect Ratio (EAR) persistence threshold tuned to trigger within **~0.35 seconds** while filtering natural ocular blinks.
* **🥱 Oral Fatigue & Yawn Frequency**: Tracks Mouth Aspect Ratio (MAR) with edge-triggered counters to prevent multiple counts for a single yawning event.
* **🧭 3D Perspective-n-Point Head Pose**: Computes head Yaw, Pitch, and Roll using canonical 3D facial geometry to flag glances away from the road.
* **📱 Edge Object Detection (ONNX)**: Quantized YOLOv8n running via `onnxruntime` CPU with strict confidence and bounding-box aspect filters to eliminate hand false-positives.
* **📈 Predictive Fatigue Engine**: 60-second rolling sliding window aggregating **PERCLOS** (Percentage of Eye Closure), yawn spikes, and gaze deviations into a 0–100% Fatigue Index.
* **🖥️ Dynamic Adaptive HUD**: Real-time vector graphics downscaled via `cv2.INTER_AREA` interpolation for sharp, legible text across any window resolution or aspect ratio.
* **🔊 Asynchronous Audio Warnings**: Stereo tone synthesized on a dedicated worker thread via Pygame to guarantee uninterrupted video capture throughput.

---

## 🛠️ System Pipeline Architecture

```text
               Webcam Stream (1280x720 Native Widescreen)
                                   │
                                   ├──> [Frame Preprocessing & Vector HUD]
                                   │
           ┌───────────────────────┴────────────────────────┐
           │                                                │
           ▼ (Every Frame)                                  ▼ (Every 2nd Frame)
MediaPipe FaceMesh (468 Points)                   YOLOv8n ONNX Engine
  ├── Eye Aspect Ratio (EAR)                        └── Cell Phone Detection
  ├── Mouth Aspect Ratio (MAR)                            (Aspect Filtered)
  └── 3D solvePnP (Yaw / Pitch)                             │
           │                                                │
           └───────────────────────┬────────────────────────┘
                                   │
                                   ▼
              Temporal Fatigue Engine (Rolling 60s Buffer)
               ├── PERCLOS % Accumulation
               ├── Yawn Frequency Tracker
               └── Inattention / Gaze Penalty
                                   │
                                   ▼
                     Decision Logic & Telemetry Dock
               ├── Asynchronous Audio Alerts (Pygame)
               └── Proportional Anti-Aliased HUD
```

---

## 📊 Detection Logic & Calibrated Thresholds

| Metric | Monitored Behavior | Trigger Threshold | Alert State |
| :--- | :--- | :--- | :--- |
| **EAR** | Acute Micro-Sleep / Closed Eyes | EAR < 0.20 sustained for >= 9 frames (~0.35s) | `DROWSINESS DETECTED` |
| **MAR** | Yawning & Drowsiness Spikes | MAR > 0.65 sustained for >= 12 frames | `YAWNING` |
| **Head Pose** | Gaze Distraction / Looking Away | \|Yaw\| > 25° or \|Pitch\| > 20° | `DISTRACTED GAZE` |
| **Object Detection** | Mobile Phone Handling | Class 67, Confidence >= 0.50, Aspect Filtered | `PHONE USE` |
| **Fatigue Index** | Cumulative Fatigue (Rolling 60s) | Score >= 60% (PERCLOS + Yawn Count + Inattention) | `FATIGUE WARNING` |
---

## 📂 Project Structure

```text
Driver-Drowsiness-Distraction-AlertEdge/
├── core/
│   ├── eye_detector.py        # EAR computation & eye state tracking
│   ├── mouth_detector.py      # MAR computation & yawn detection
│   ├── head_pose.py           # 3D solvePnP pose estimation (Yaw/Pitch/Roll)
│   ├── object_detector.py     # YOLOv8n ONNX inference & geometry filter
│   └── fatigue_scorer.py      # Rolling 60s temporal PERCLOS & fatigue engine
├── models/
│   └── yolov8n.onnx           # Lightweight ONNX detection model (~12.3 MB)
├── utils/
│   └── sound_alert.py         # Multi-threaded audio tone generator
├── requirements.txt           # Pinned production dependencies
├── main.py                    # Main pipeline, arbitration & vector HUD
├── run.bat                    # One-click Windows runner
└── README.md                  # Project documentation
```
---

## 🚀 Getting Started

### Prerequisites

* Python 3.10 or 3.11
* Integrated or external USB Webcam
* Windows 10/11 (or Linux/macOS)

---

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
    git clone https://github.com/AreebaAminn/Driver-Drowsiness-Distraction-AlertEdge.git
    cd Driver-Drowsiness-Distraction-AlertEdge
   ```

2. **Set Up a Virtual Environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

---

## 💻 Running the Monitor

* **Option A: Command Line**
  ```powershell
  python main.py
  ```

* **Option B: One-Click Execution (Windows)**  
  Double-click `run.bat` in the project root.

> **Key Controls**: Press **`q`** while focused on the display window to cleanly release camera resources and terminate the session.

---

## ⚖️ License

Distributed under the MIT License. See `LICENSE` for more information.
