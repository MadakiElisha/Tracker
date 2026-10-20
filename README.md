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


<img width="1873" height="927" alt="drone_trackerII" src="https://github.com/user-attachments/assets/6d00be7b-15e3-4a9a-a61e-8286ef22a0bd" />

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
├── tracker.py             # Main tracker (run this)
├── config.yaml            # ALL tunable parameters (PID, safety, classes, FC, GCS)
├── pid_controller2.py     # 3-axis PID (config-driven, anti-windup, range-based distance)
├── safety.py              # Failsafe supervisor (video/link loss, pilot override, battery)
├── fc_interface5.py       # FC layer (ArduPilot; PX4 not implemented)
├── tracker_state.py       # Shared thread-safe state
├── gcs_server.py          # Browser view + tap-to-lock (token-protected)
├── mouse_lock.py          # OpenCV click-to-lock
├── gz_camera.py           # Gazebo camera source (--source gz:/drone/camera)
├── run_logger.py          # Per-frame CSV run log
├── analyze_run.py         # PASS/FAIL scoring of a run log
├── model_download.py      # Fetches the VisDrone YOLO11n weights (best.pt)
├── tracker_logger.py      # Logging helpers
├── inference_backend.py   # Hardware auto-detection (Jetson/RPi5/PC)
├── tests/                 # 172 tests: pytest tests
├── sim/                   # launch_sim.sh, move_target.py, add_camera_to_iris.py, worlds/, params/
├── scripts/setup_python_env.sh
├── deployment/            # TensorRT / Hailo export
├── docs/                  # SETUP_UBUNTU_24_04.md, SIM_VALIDATION.md, Jetson & RPi5 guides
├── legacy/                # Earlier iterations kept for reference
└── requirements*.txt
```

---

## Quick Start

Full step-by-step for a fresh Ubuntu 24.04 machine: **[docs/SETUP_UBUNTU_24_04.md](docs/SETUP_UBUNTU_24_04.md)**.

```bash
git clone https://github.com/MadakiElisha/Tracker.git && cd Tracker
bash scripts/setup_python_env.sh && source venv/bin/activate   # venv, deps, model, tests
python3 tracker.py --no-fly --source /path/to/video.mp4         # no simulator needed
```

Full simulation (after installing Gazebo Harmonic, ArduPilot SITL and ardupilot_gazebo):

```bash
python3 sim/add_camera_to_iris.py ~/ardupilot_gazebo      # once
cd sim && ./launch_sim.sh ardupilot                       # terminal 1
python3 sim/move_target.py                                # terminal 2
python3 tracker.py --connect 127.0.0.1:14550 --source gz:/drone/camera   # terminal 3
```

| Key / action | Result |
|---|---|
| Click a box / `L` | Lock target |
| `R` | Release lock |
| `T` | Arm and take off |
| `H` | Return to launch |
| `K` | Land |
| `+` / `-` | Follow distance |
| `Q` | Quit (RTL if airborne) |
| `rc 7 2000` / `rc 7 1000` in MAVProxy | RC lock / release |
| `rc 9 1000/1500/2000` | Filter Any / Person / Vehicle |
| Tap video at http://localhost:8080 | Browser lock (set `gcs.token`) |

Validation: `docs/SIM_VALIDATION.md`. Tests: `python3 -m pytest tests -q`.
Status: unit and wiring tests pass; the live Gazebo closed loop and real hardware are NOT yet validated.

---

## Flight Controller Compatibility

| FC Firmware | Protocol | Connection | Status |
|-------------|----------|------------|--------|
| ArduPilot | MAVLink 2.0 | UART / UDP | ✅ Full support |
| PX4 | MAVLink 2.0 | UART / UDP | ❌ Not implemented |
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
- **Model:** YOLO11n fine-tuned on VisDrone (`dronefreak/visdrone-yolov11n`, AGPL-3.0)
- **Classes (VisDrone IDs):** pedestrian 0, people 1, car 3, van 4, truck 5, bus 8, motor 9
- **Confidence threshold / resolution:** see `config.yaml` (default 0.35, 640x480)

### Tracking
- **Tracker:** ByteTrack via Ultralytics; lock dropped after `tracking.max_lost_frames`

### Control
- All gains and limits live in `config.yaml` (`pid:`, `flight:`, `safety:`)
- Distance uses relative range error with per-class setpoints; forward motion is gated near frame edges
- Safety supervisor: video loss, link loss, pilot mode change, battery -> hover / RTL / release

### FC Interface
- Pure pymavlink (no DroneKit dependency)
- Queue-based command architecture — latest command always wins
- Single telemetry thread handles HEARTBEAT, GPS, battery, and RC channels
- Correct autopilot component targeting (waits for component=1 specifically)

---

## Known Limitations

- Occlusion handling is Kalman-based prediction only — no appearance Re-ID yet
- iNav and Betaflight MSP backends not yet implemented
- GCS runs on waitress; set a token and keep it off untrusted networks
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

See `requirements.txt` (PC), `requirements_jetson.txt`, `requirements_rpi5.txt`. Python 3.12 on Ubuntu 24.04.
System packages: `python3-venv libgl1 netcat-openbsd`, and for the simulator `gz-harmonic`,
`python3-gz-transport13`, `python3-gz-msgs10`, ArduPilot SITL, ardupilot_gazebo.

---

## Licence

MIT for this code. NOTE: the VisDrone model weights are AGPL-3.0; review the combined licence before distributing.

---

## Acknowledgements

Built with ArduPilot SITL, Gazebo Harmonic, Ultralytics YOLO,
ByteTrack, and pymavlink. Inspired by the capabilities of
Skydio X10 and Anduril Bolt.

