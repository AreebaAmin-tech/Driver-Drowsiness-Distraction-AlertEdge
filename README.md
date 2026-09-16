Markdown# Edge Driver Drowsiness & Distraction Monitoring System

A real-time, lightweight computer vision safety pipeline designed for Windows edge devices. The system tracks driver vigilance, micro-sleep events, yawning fatigue, head gaze deviation, and mobile phone distraction with minimal CPU overhead and zero frame lag.

---

## Key Features

* **Instant Micro-Sleep Detection**: Eye Aspect Ratio (EAR) evaluation calibrated for fast detection (~0.35s) while filtering natural blinks.
* **Yawn & Fatigue Tracking**: Mouth Aspect Ratio (MAR) edge detector tracking individual yawn frequency over a rolling window.
* **3D Head Pose Estimation**: Perspective-n-Point (`solvePnP`) using canonical 3D facial landmarks to detect extreme yaw (looking away) and pitch (nodding off).
* **Edge Object Detection**: YOLOv8n running via `onnxruntime` CPU with letterboxed aspect ratio normalization and geometry filtering to eliminate bare-hand false positives.
* **Cumulative Fatigue Engine**: 60-second rolling temporal window computing **PERCLOS**, yawn accumulation, and distraction penalties into a real-time 0–100% Fatigue Index.
* **Responsive Vector HUD**: Dynamically scaled bottom dock telemetry rendered with high-quality area downsampling (`cv2.INTER_AREA`) to prevent text pixelation at any window scale.
* **Non-Blocking Stereo Alerts**: Synthesized multi-threaded audio alerts via `pygame` that never interrupt frame throughput.

---

## System Architecture

```text
 Webcam Stream (1280x720 Native Widescreen)
                  │
                  ├──> [Frame Preprocessing & Quality Downsampling]
                  │
 ┌────────────────┴────────────────────────┐
 │                                         │
 ▼ (Every Frame)                           ▼ (Every 2nd Frame)
MediaPipe FaceMesh (468 Keypoints)       YOLOv8n ONNX Engine (onnxruntime)
 │                                         │
 ├──> Eye Aspect Ratio (EAR)               └──> Mobile Phone Bounding Boxes
 ├──> Mouth Aspect Ratio (MAR)                  (Aspect-ratio filtered)
 └──> 3D solvePnP (Yaw / Pitch)
 │                                         │
 └────────────────┬────────────────────────┘
                  │
                  ▼
        Temporal Fatigue Scorer (Rolling 60s Window)
         ├── PERCLOS % Accumulation
         ├── Yawn Frequency Tracker
         └── Distraction Time Penalty
                  │
                  ▼
        Arbitration & HUD Rendering
         ├── Asynchronous Alert Trigger (pygame stereo tone)
         └── Dynamic Bottom Dock Overlay (EAR, MAR, Pose, Fatigue Bar)
Directory StructurePlaintextDriver Drowsiness & Distraction AlertEdge/
│
├── core/
│   ├── eye_detector.py      # EAR computation & blink persistence
│   ├── mouth_detector.py    # MAR yawn analysis
│   ├── head_pose.py         # 3D solvePnP rotation & translation engine
│   ├── object_detector.py   # YOLOv8n ONNX inference & box verification
│   └── fatigue_scorer.py    # Rolling 60s PERCLOS and fatigue accumulator
│
├── models/
│   └── yolov8n.onnx         # Exported ONNX weights (~12.3 MB)
│
├── utils/
│   └── sound_alert.py       # Asynchronous stereo audio synthesizer
│
├── requirements.txt         # Pinned dependency manifest
├── main.py                  # Real-time pipeline, arbitration, dynamic HUD
├── run.bat                  # One-click Windows execution script
└── README.md                # Project documentation
Installation & Setup1. Clone the RepositoryBashgit clone https://github.com/AreebaAmin-tech/Driver-Drowsiness-Distraction-AlertEdge.git
cd Driver-Drowsiness-Distraction-AlertEdge
2. Set Up Virtual EnvironmentPowerShellpython -m venv venv
.\venv\Scripts\Activate.ps1
3. Install DependenciesPowerShellpip install -r requirements.txt
Running the ApplicationOption A: Via Command LinePowerShellpython main.py
Option B: One-Click Windows LaunchDouble-click run.bat in File Explorer.Press q in the camera window to safely stop video acquisition and exit.Telemetry & Thresholds ReferenceMetricThreshold / ConditionTrigger OutputEAR< 0.20 sustained for $\ge 9$ frames (~0.35s)DROWSINESS DETECTEDMAR> 0.65 sustained for $\ge 12$ framesYAWNINGHead Pose$\|\text{Yaw}\| > 25^\ci$ or $ $\|\text{Pitch}\| > 20^\$DISTRACTED GAZEMobile PhoneYOLOv8n Confidence $\ge 0.50$ + Box Aspect SanityPHONE USEFatigue Index$\ge 60\%$ (Accumulated PERCLOS + Yawns + Distraction)FATIGUE WARNING