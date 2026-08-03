<div align="center">

# 🎯 Drone Tracker
### Autonomous Multi-Class Target Detection, Locking, and Follow System

*Inspired by Skydio X10 and Anduril Bolt — built from scratch.*

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![YOLO](https://img.shields.io/badge/YOLO-v11n-darkgreen)
![ArduPilot](https://img.shields.io/badge/ArduPilot-4.x-red)
![Platform](https://img.shields.io/badge/Platform-Jetson%20%7C%20RPi5%20%7C%20PC-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

---

## What This Is

A complete autonomous drone follow-me system that detects, locks on to, and follows
any designated target — persons, cars, motorcycles, buses, or trucks — with high
precision and occlusion resilience. Built to run on a companion computer aboard an
FPV drone, interfacing directly with ArduPilot, PX4, iNav, or Betaflight flight
controllers via MAVLink or MSP.

This is not a toy tracker. It is a full perception-to-control pipeline designed
for real deployment.

Heads-up: You will find quite some duplicate files here and there. Those came about during dev. May come in handy for you.
---

## Key Features

- **Multi-class detection** — persons, cars, motorcycles, buses, trucks simultaneously
- **Persistent track locking** — ByteTrack assigns stable IDs across frames
- **3-axis PID control** — yaw, forward/back, and altitude all respond to target position
- **Occlusion handling** — Kalman filter prediction keeps tracking through temporary disappearances
- **Three target assignment methods** — RC switch (CH7), GCS browser tap-to-lock, OpenCV window click
- **CH9 class filter** — 3-position RC switch cycles between Person / Vehicle / Any
- **FC-agnostic architecture** — single abstraction layer supports ArduPilot, PX4, iNav, Betaflight
- **Watchdog failsafe** — zero-velocity hover triggered automatically if tracker stream dies
- **Browser GCS** — live annotated video + tap-to-lock from any device on the network
- **Hardware auto-detection** — switches between TensorRT (Jetson) / HailoRT (RPi5) / PyTorch (PC)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     COMPANION COMPUTER                           │
│                                                                  │
│  Camera Feed                                                     │
│      │                                                           │
│      ▼                                                           │
│  YOLO11n Detector ──────────────────────────────────────┐       │
│  (TensorRT / HailoRT / PyTorch)                         │       │
│      │                                                  │       │
│      ▼                                                  ▼       │
│  ByteTrack Multi-Object Tracker          GCS Browser (Flask)    │
│      │                                  Live feed + tap-to-lock │
│      ▼                                                  │       │
│  Target Selector ◄──────────────────────────────────────┘       │
│  (RC CH7 switch │ GCS tap │ Mouse click)                        │
│      │                                                           │
│      ▼                                                           │
│  3-Axis PID Controller                                           │
│  (Yaw + Forward/Back + Altitude)                                 │
│      │                                                           │
│      ▼                                                           │
│  FC Abstraction Layer                                            │
│  (ArduPilot │ PX4 │ iNav │ Betaflight)                          │
│      │                                                           │
└──────┼──────────────────────────────────────────────────────────┘
       │ MAVLink (UART or UDP)
       ▼
  Flight Controller → Motors
```

---

## Hardware Targets

| Platform | Inference | FPS | Latency |
|----------|-----------|-----|---------|
| Dev PC (SITL) | PyTorch CPU/CUDA | 10-80 FPS | varies |
| Jetson Orin Nano | TensorRT FP16 | 80-120 FPS | 8-12ms |
| Jetson Orin NX | TensorRT FP16 | 120-180 FPS | 5-8ms |
| RPi 5 + Hailo-8L | HailoRT INT8 | 30-45 FPS | 22-33ms |

---

## Repository Structure

```
Tracker/
│
├── tracker2bi.py          # Main integrated tracker (run this)
├── pid_controller2.py     # 3-axis PID controller (yaw, pitch, Z)
├── fc_interface5.py       # FC abstraction layer (ArduPilot / PX4)
├── perception.py          # Standalone perception pipeline (dev/test)
├── gcs_server.py          # Browser GCS — MJPEG stream + tap-to-lock
├── mouse_lock.py          # OpenCV window click-to-lock handler
├── connection_test.py     # FC connection and takeoff test
├── inference_backend.py   # Hardware auto-detection (Jetson/RPi5/PC)
│
├── sim/                   # Simulation environment
│   ├── launch_sim.sh      # Starts Gazebo + ArduPilot SITL
│   ├── move_target.py     # Moves simulated targets in Gazebo
│   ├── worlds/
│   │   └── tracker_world.sdf   # Gazebo world (drone + targets + wall)
│   └── params/
│       └── ardupilot_tracker.param  # ArduPilot SITL parameters
│
├── deployment/            # Hardware deployment
│   ├── export_tensorrt.py # Export YOLO11n → TensorRT (run on Jetson)
│   ├── export_hailo.py    # Export YOLO11n → Hailo HEF (run on dev PC)
│   └── start.sh           # Auto-detecting startup script
│
├── docs/
│   ├── PHASE_6A_JETSON.md      # Complete Jetson deployment guide
│   └── PHASE_6B_RPI5_HAILO.md  # Complete RPi 5 + Hailo-8L guide
│
├── requirements.txt            # Dev / PC dependencies
├── requirements_jetson.txt     # Jetson-specific dependencies
├── requirements_rpi5.txt       # RPi 5 + Hailo-8L dependencies
└── .gitignore
and some duplicates
```

---

## Quick Start — Simulation (SITL)

### Prerequisites

- Ubuntu 22.04 or 24.04 (native or WSL2)
- Python 3.10+
- ArduPilot SITL ([install guide](https://ardupilot.org/dev/docs/SITL-setup-landvehicles.html))
- Gazebo Harmonic ([install guide](https://gazebosim.org/docs/harmonic/install))
- ardupilot_gazebo plugin ([repo](https://github.com/ArduPilot/ardupilot_gazebo))

### 1. Clone and install dependencies

```bash
git clone https://github.com/madakielisha/Tracker.git
cd Tracker

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the simulation

```bash
# Terminal 1 — launch Gazebo + ArduPilot SITL
cd sim && ./launch_sim.sh ardupilot
# Wait for: STABILIZE> and "ALL SYSTEMS RUNNING"

# Terminal 2 — start moving targets in the simulation
source ../venv/bin/activate
python3 sim/move_target.py

# Terminal 3 — run the tracker
source venv/bin/activate
python3 tracker2bi.py \
    --fc ardupilot \
    --connect 127.0.0.1:14550 \
    --source path/to/test_video.mp4 \
    --no-fly
```

### 3. Test target locking

| Action | Result |
|--------|--------|
| Click bounding box in OpenCV window | Lock on that specific target |
| Press `L` | Lock on target |
| Press `R` | Release lock |
| Press `T` | Arm and takeoff (requires SITL running) |
| Press `H` | Return to home (RTL) |
| Press `+` / `-` | Increase / decrease follow distance |
| Press `Q` | Quit and RTL |
| `rc 7 2000` in MAVProxy | RC switch lock (CH7) |
| `rc 7 1000` in MAVProxy | RC switch release (CH7) |
| `rc 9 1000/1500/2000` | Filter: Any / Person / Vehicle |
| Tap video in browser | GCS tap-to-lock |

### 4. Open the GCS browser

```
http://localhost:8080
```

Or from Windows/another device:
```
http://<WSL2-IP>:8080
```

---

## Full Flight Run (with SITL flying)

```bash
# After SITL is running and tracker is started:
# In the OpenCV window or MAVProxy:

# 1. Take off
# Press T  (or in MAVProxy: mode guided → arm throttle → takeoff 10)

# 2. Lock a target
# Click any bounding box in the video window

# 3. Watch the drone follow in Gazebo
# Vx, Vz, Yaw values in HUD will be non-zero
# Drone in Gazebo rotates and moves toward/with the target

# 4. Test occlusion
# Target will pass behind the wall in the world
# Drone coasts, then re-acquires when target re-emerges

# 5. Return home
# Press H or flip CH8 on transmitter
```

---

## Flight Controller Compatibility

| FC Firmware | Protocol | Connection | Status |
|-------------|----------|------------|--------|
| ArduPilot | MAVLink 2.0 | UART / UDP | ✅ Full support |
| PX4 | MAVLink 2.0 | UART / UDP | ✅ Full support |
| iNav | MSP | UART | 🔄 |
| Betaflight | MSP | UART | 🔄 |

---

## RC Channel Mapping

| Channel | Switch Type | Function |
|---------|-------------|----------|
| CH7 | 2-position momentary | Lock on largest visible target |
| CH7 | 2-position toggle | Release lock |
| CH9 | 3-position switch | Class filter: Any / Person / Vehicle |

In SITL, simulate with MAVProxy: `rc 7 2000`, `rc 7 1000`, `rc 9 1500`

---

## Hardware Deployment

See full deployment guides in `/docs`:

- **[Jetson Orin Nano/NX →](docs/PHASE_6A_JETSON.md)** JetPack setup, TensorRT export,
  CSI camera, UART FC, autostart
- **[RPi 5 + Hailo-8L →](docs/PHASE_6B_RPI5_HAILO.md)** Hailo driver, HEF export,
  Picamera2, UART FC, autostart

### One-command hardware detection

```python
from inference_backend import detect_platform, open_camera
model_path, platform = detect_platform()
# Automatically returns:
# yolo11n.engine on Jetson  (TensorRT)
# yolo11n.hef    on RPi5    (HailoRT)
# yolo11n.pt     on dev PC  (PyTorch)
```

### Export model for deployment

```bash
# On Jetson (TensorRT):
python3 deployment/export_tensorrt.py

# On dev PC or RPi (Hailo HEF):
python3 deployment/export_hailo.py
```

---

## Architecture Details

### Detection
- **Model:** YOLO11n (Ultralytics)
- **Classes tracked:** Person (0), Car (2), Motorcycle (3), Bus (5), Truck (7)
- **Confidence threshold:** 0.45
- **Input resolution:** 640×480

### Tracking
- **Tracker:** ByteTrack (via boxmot)
- **Occlusion tolerance:** 60 frames (~2-5 seconds depending on FPS)
- **Kalman filter:** built into ByteTrack, predicts position during occlusion
- **Re-ID warning:** displayed when confidence < 0.6 on re-acquired track

### Control
- **Yaw PID:** centres target horizontally (Kp=1.2, Ki=0.02, Kd=0.15)
- **Pitch PID:** maintains follow distance via area estimation (Kp=8.0, Ki=0.05, Kd=1.5)
- **Z PID:** maintains vertical centering of target in frame (Kp=1.5, Ki=0.01, Kd=0.2)
- **Smoothing:** exponential low-pass filter (α=0.4) on all command outputs
- **Watchdog:** 0.5s timeout forces zero-velocity hover if tracker stream dies

### FC Interface
- Pure pymavlink (no DroneKit dependency)
- Queue-based command architecture — latest command always wins
- Single telemetry thread handles HEARTBEAT, GPS, battery, and RC channels
- Correct autopilot component targeting (waits for component=1 specifically)

---

## Known Limitations

- Occlusion handling is Kalman-based prediction only — no appearance Re-ID yet
- iNav and Betaflight MSP backends not yet implemented
- GCS browser uses Flask development server — production deployment needs gunicorn
- TensorRT engines are hardware-specific — rebuild on each target device

---

## Roadmap

- Simulation environment (Gazebo + ArduPilot SITL)
- Perception pipeline (YOLO11n + ByteTrack)
- FC abstraction layer (ArduPilot + PX4)
- 3-axis PID controller
- Target assignment (RC switch + GCS tap + mouse click)
- Hardware deployment guides (Jetson + RPi5 + Hailo)
- iNav / Betaflight MSP RC injection
- Appearance-based Re-ID for full occlusion recovery
- OSD text injection to FPV goggles
- HDZero bounding box overlay on FPV feed

---

## Dependencies

| Package | Purpose |
|---------|---------|
| ultralytics | YOLO11n detection and tracking |
| boxmot | ByteTrack multi-object tracker |
| pymavlink | MAVLink FC communication |
| opencv-python | Video capture and display |
| flask | GCS browser server |
| scipy | Kalman filter utilities |
| pyserial | Serial UART communication |

---

## Licence

MIT — do whatever you want with this, just don't use it to harm people.

---

## Acknowledgements

Built with ArduPilot SITL, Gazebo Harmonic, Ultralytics YOLO,
ByteTrack, and pymavlink. Inspired by the capabilities of
Skydio X10 and Anduril Bolt.
