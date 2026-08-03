# Phase 6A — Jetson Orin Nano / Orin NX Deployment Guide
# Drone Tracker — Full Hardware Deployment

## Hardware Required
- Jetson Orin Nano 8GB Developer Kit  OR  Jetson Orin NX 8GB/16GB Module + Carrier Board
- MicroSD card (64GB+) or NVMe SSD (recommended for NX)
- Arducam IMX477 Global Shutter Camera (CSI interface)
- USB-to-UART adapter (for FC connection during bench testing)
- 5V/4A USB-C power supply (developer kit) or regulated 5V from drone PDB

---

## STEP 1 — Flash JetPack OS

### 1.1 Download SDK Manager (on a Ubuntu 20.04 or 22.04 host PC)
JetPack must be flashed from a Linux host. WSL2 on Windows does NOT work
for flashing — use a native Ubuntu machine or a VM with USB passthrough.

Download from:
  https://developer.nvidia.com/sdk-manager

Install it:
  sudo dpkg -i sdkmanager_<version>_amd64.deb

### 1.2 Flash JetPack 6.x
1. Connect Jetson to host via USB-C
2. Put Jetson into recovery mode:
   - Hold FORCE RECOVERY button
   - Press and release RESET button
   - Release FORCE RECOVERY after 2 seconds
3. Run SDK Manager and select:
   - Product: Jetson
   - Hardware: Jetson Orin Nano (or NX)
   - JetPack version: 6.x (latest stable)
   - Components: Jetson Linux + Jetson Runtime Components
4. Follow prompts. Flash takes ~20 minutes.

### 1.3 First boot setup
- Connect monitor, keyboard, USB hub to Jetson
- Boot and complete Ubuntu 22.04 initial setup wizard
- Set username: drone
- Set hostname: dronetracker
- Enable SSH:
    sudo systemctl enable ssh
    sudo systemctl start ssh

### 1.4 Verify JetPack version
    sudo apt show nvidia-jetpack
    # Should show: Version: 6.x.x

### 1.5 Verify CUDA is present
    nvcc --version
    # Should show: release 12.x

    python3 -c "import torch; print(torch.cuda.is_available())"
    # Should print: True
    # Note: PyTorch for Jetson comes pre-installed in JetPack 6


---

## STEP 2 — System Configuration

### 2.1 Set maximum performance mode
    sudo nvpmodel -m 0          # MAX power mode
    sudo jetson_clocks           # Lock all clocks to max

    # Verify mode
    sudo nvpmodel -q
    # Should show: NV Power Mode: MAXN

### 2.2 Set performance mode to persist across reboots
    # Create a systemd service that sets max clocks on boot
    sudo tee /etc/systemd/system/jetson-perf.service << 'EOF'
    [Unit]
    Description=Jetson Maximum Performance Mode
    After=multi-user.target

    [Service]
    Type=oneshot
    ExecStart=/usr/bin/nvpmodel -m 0
    ExecStart=/usr/bin/jetson_clocks
    RemainAfterExit=yes

    [Install]
    WantedBy=multi-user.target
    EOF

    sudo systemctl enable jetson-perf.service

### 2.3 Configure UART for FC connection
The Jetson Orin Nano has UART on the 40-pin header:
  - Pin 8  (TXD) → FC RX
  - Pin 10 (RXD) → FC TX
  - Pin 6  (GND) → FC GND

    # Enable UART
    sudo systemctl stop nvgetty
    sudo systemctl disable nvgetty
    sudo systemctl mask nvgetty

    # Add drone user to dialout group
    sudo usermod -a -G dialout drone

    # Verify UART device exists
    ls /dev/ttyTHS0   # or ttyTHS1 depending on carrier board

### 2.4 Disable unnecessary services (reduces boot time and RAM usage)
    sudo systemctl disable bluetooth
    sudo systemctl disable cups
    sudo systemctl disable avahi-daemon
    sudo systemctl disable ModemManager


---

## STEP 3 — Install Project Dependencies

### 3.1 System packages
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        libopencv-dev \
        python3-opencv \
        git \
        cmake \
        build-essential \
        v4l-utils \
        libgstreamer1.0-dev \
        gstreamer1.0-tools \
        gstreamer1.0-plugins-good \
        gstreamer1.0-plugins-bad \
        portaudio19-dev

### 3.2 Create project virtual environment
NOTE: On Jetson, we use --system-site-packages to inherit
the pre-built JetPack PyTorch and CUDA bindings.
Building PyTorch from scratch on Jetson takes hours.

    cd ~
    git clone <your drone_tracker repo>   # or scp from dev machine
    cd drone_tracker

    python3 -m venv venv --system-site-packages
    source venv/bin/activate

### 3.3 Install Python packages
    pip install --upgrade pip

    # Core tracker packages
    pip install \
        pymavlink \
        pyserial \
        flask \
        supervision \
        scipy

    # Ultralytics — use the Jetson-compatible wheel
    pip install ultralytics

    # BoxMOT for ByteTrack
    pip install boxmot

    # Verify CUDA is visible inside venv
    python3 -c "
    import torch
    print('CUDA available:', torch.cuda.is_available())
    print('Device:', torch.cuda.get_device_name(0))
    "
    # Expected: CUDA available: True
    #           Device: Orin


---

## STEP 4 — TensorRT Engine Export

This is the core performance step. We convert YOLO11n from PyTorch (.pt)
to a TensorRT engine (.engine) that runs natively on the Jetson GPU.
Inference drops from ~80ms (PyTorch) to ~8ms (TensorRT). 4-10x speedup.

IMPORTANT: TensorRT engines are hardware-specific. An engine built on
Jetson Orin Nano will NOT run on a different GPU. Always build the engine
on the target hardware.

### 4.1 Export the engine (run ON the Jetson, not on dev PC)
    source ~/drone_tracker/venv/bin/activate
    cd ~/drone_tracker

    python3 << 'PYEOF'
from ultralytics import YOLO

print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

print("Exporting to TensorRT engine...")
print("This takes 5-15 minutes on first run. Do not interrupt.")
model.export(
    format      = "engine",
    device      = 0,           # GPU 0
    half        = True,        # FP16 — halves memory, ~same accuracy
    imgsz       = 640,         # must match inference resolution
    workspace   = 4,           # GB of GPU RAM for TRT optimization
    verbose     = True,
)
print("Export complete: yolo11n.engine")
PYEOF

### 4.2 Verify the engine loads and runs
    python3 << 'PYEOF'
from ultralytics import YOLO
import cv2, time, numpy as np

print("Loading TensorRT engine...")
model = YOLO("yolo11n.engine", task="detect")

# Warm up (first inference is always slow due to TRT initialization)
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
print("Warming up engine (3 passes)...")
for i in range(3):
    _ = model(dummy, verbose=False)

# Benchmark
print("Benchmarking 100 frames...")
t_start = time.time()
for i in range(100):
    _ = model(dummy, verbose=False)
elapsed = time.time() - t_start

print(f"100 frames in {elapsed:.2f}s")
print(f"Throughput: {100/elapsed:.1f} FPS")
print(f"Latency:    {elapsed/100*1000:.1f}ms per frame")
PYEOF
    # Expected: 80-120 FPS, 8-12ms latency on Orin Nano
    # Expected: 120-180 FPS, 5-8ms latency on Orin NX

### 4.3 Update YOLO_MODEL path in tracker
    # In tracker2bi.py, change:
    YOLO_MODEL = "yolo11n.pt"
    # To:
    YOLO_MODEL = "yolo11n.engine"
    # The rest of the code is identical — Ultralytics handles TRT transparently


---

## STEP 5 — Camera Integration (Arducam IMX477 Global Shutter)

Global shutter is critical for FPV drones. Rolling shutter cameras (like the
standard RPi HQ camera) produce jello/wobble artifacts at speed which
destroy detection quality. IMX477 is a global shutter sensor.

### 5.1 Connect the camera
- Connect IMX477 to CAM0 CSI port on Jetson carrier board
- Use the CSI ribbon cable that came with the Arducam module
- Lock the ribbon cable connector carefully

### 5.2 Verify camera is detected
    # Check V4L2 devices
    v4l2-ctl --list-devices
    # Should show: /dev/video0

    # Test raw capture
    v4l2-ctl --device=/dev/video0 \
              --set-fmt-video=width=1280,height=720,pixelformat=MJPG \
              --stream-mmap --stream-count=1 \
              --stream-to=/tmp/test_frame.jpg
    # No errors = camera is working

### 5.3 GStreamer pipeline for CSI camera in OpenCV
Standard cv2.VideoCapture(0) works but is slow via V4L2.
Use GStreamer pipeline for hardware-accelerated capture:

    GSTREAMER_PIPELINE = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, "
        "framerate=60/1, format=NV12 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink drop=1"
    )
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    # In tracker2bi.py, replace the VideoCapture line with this
    # when running on Jetson with CSI camera.
    # The pipeline: captures at 1280x720@60fps, downscales to 640x480 in GPU

### 5.4 Test the full camera pipeline
    python3 << 'PYEOF'
import cv2

PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "framerate=60/1, format=NV12 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! "
    "appsink drop=1"
)

cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
if not cap.isOpened():
    print("ERROR: Cannot open CSI camera. Check cable connection.")
    exit(1)

print("Camera open. Press Q to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed.")
        break
    cv2.imshow("CSI Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
PYEOF


---

## STEP 6 — MAVLink Over UART to Flight Controller

### 6.1 Wiring
Jetson 40-pin header UART0:
  Pin 8  (TXD0) ────► FC UART RX
  Pin 10 (RXD0) ◄──── FC UART TX
  Pin 6  (GND)  ────── FC GND
  Do NOT connect 5V — Jetson and FC have separate power rails.

ArduPilot FC settings (set in Mission Planner or QGC):
  SERIAL1_BAUD   = 921 (921600 baud)
  SERIAL1_PROTOCOL = 2 (MAVLink 2)
  BRD_SER1_RTSCTS = 0 (no flow control)

### 6.2 Test UART MAVLink connection
    # Update connection string in tracker
    # Change: '127.0.0.1:14550'
    # To:     '/dev/ttyTHS0'
    #
    # In fc_interface5.py _connect_impl:
    # self.master = mavutil.mavlink_connection('/dev/ttyTHS0', baud=921600)
    #
    # Run connection test:
    python3 fc_interface5.py

    # Expected:
    # [ARDUPILOT] Awaiting autopilot heartbeat (component 1)...
    # ...saw heartbeat from system=1 component=1
    # [ARDUPILOT] Connected to System: 1 Component: 1


---

## STEP 7 — Platform Detection and Runtime Backend Selection

This makes the tracker automatically pick TensorRT on Jetson,
HailoRT on RPi 5, and standard PyTorch on dev/PC.
Add this to tracker2bi.py:

    def detect_inference_backend():
        """Detect hardware and return appropriate model path and backend."""

        # Check for Jetson (CUDA available + Jetson-specific nvpmodel)
        import os, subprocess
        is_jetson = os.path.exists('/usr/bin/nvpmodel')
        if is_jetson:
            engine_path = 'yolo11n.engine'
            if os.path.exists(engine_path):
                print(f"[BACKEND] Jetson detected → TensorRT engine")
                return engine_path, 'tensorrt'
            else:
                print(f"[BACKEND] Jetson detected but no .engine found.")
                print(f"  Run: python3 export_engine.py first")
                print(f"  Falling back to PyTorch (slow).")
                return 'yolo11n.pt', 'pytorch'

        # Check for Hailo (RPi 5 + Hailo-8L)
        hailo_dev = '/dev/hailo0'
        if os.path.exists(hailo_dev):
            hef_path = 'yolo11n.hef'
            if os.path.exists(hef_path):
                print(f"[BACKEND] Hailo-8L detected → HailoRT engine")
                return hef_path, 'hailo'
            else:
                print(f"[BACKEND] Hailo-8L detected but no .hef found.")
                print(f"  Run: python3 export_hailo.py first")
                print(f"  Falling back to PyTorch (slow).")
                return 'yolo11n.pt', 'pytorch'

        # Default: standard PyTorch CPU/CUDA
        print(f"[BACKEND] Standard platform → PyTorch")
        return 'yolo11n.pt', 'pytorch'


---

## STEP 8 — Autostart on Boot (systemd service)

The tracker starts automatically when the Jetson powers on.
No keyboard, no monitor, no manual launch needed.

### 8.1 Create the service file
    sudo tee /etc/systemd/system/drone-tracker.service << 'EOF'
    [Unit]
    Description=Drone Tracker Autonomous Follow System
    After=network.target nvpmodel.service
    Wants=network.target

    [Service]
    Type=simple
    User=drone
    WorkingDirectory=/home/drone/drone_tracker
    Environment=DISPLAY=:0
    Environment=PYTHONUNBUFFERED=1
    ExecStartPre=/usr/bin/nvpmodel -m 0
    ExecStartPre=/usr/bin/jetson_clocks
    ExecStart=/home/drone/drone_tracker/venv/bin/python3 \
        tracker2bi.py \
        --fc ardupilot \
        --connect /dev/ttyTHS0 \
        --no-display
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    EOF

    sudo systemctl daemon-reload
    sudo systemctl enable drone-tracker.service

### 8.2 Add --no-display flag to tracker
When running headless (no monitor), OpenCV imshow crashes.
Add this to tracker2bi.py argument parser:

    parser.add_argument('--no-display', action='store_true',
        help='Disable OpenCV window — headless/autostart mode')

And in the main loop, gate the imshow:

    if not args.no_display:
        cv2.imshow("Drone Tracker", frame)
        cv2.setMouseCallback("Drone Tracker", mouse_locker.handle_click)

    # GCS browser is always active — primary interface in headless mode
    gcs.update_frame(frame)

### 8.3 Start/stop/status commands
    sudo systemctl start drone-tracker       # start now
    sudo systemctl stop drone-tracker        # stop
    sudo systemctl status drone-tracker      # view status
    journalctl -u drone-tracker -f           # live log output


---

## STEP 9 — Thermal and Power Management

FPV drones are hot, enclosed environments. Jetson will throttle
if it hits 80°C — this causes FPS drops mid-flight.

### 9.1 Monitor temperature during bench testing
    # Watch temperatures in real time
    watch -n 1 cat /sys/devices/virtual/thermal/thermal_zone*/temp

    # Or use tegrastats
    tegrastats --interval 500

### 9.2 Add fan control (if using a cooling fan)
    # Most Jetson carrier boards with a fan use:
    sudo sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'
    # 255 = full speed, 128 = half speed

    # Set fan to max in the autostart service ExecStartPre:
    ExecStartPre=/bin/sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'

### 9.3 Physical mounting considerations
- Mount Jetson module away from ESCs and motor wires
- Ensure airflow over the heatsink
- Use copper standoffs not nylon for heat dissipation to frame
- Never enclose in a sealed compartment


---

## STEP 10 — Full System Verification Checklist

Run this on the Jetson before first flight:

    python3 << 'PYEOF'
import sys, os, subprocess

checks = []

# CUDA
try:
    import torch
    checks.append(("CUDA available",
                   torch.cuda.is_available(),
                   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"))
except: checks.append(("CUDA", False, "torch not found"))

# TensorRT engine
engine = "yolo11n.engine"
checks.append(("TensorRT engine", os.path.exists(engine), engine))

# Camera
cam_ok = os.path.exists('/dev/video0')
checks.append(("Camera device", cam_ok, "/dev/video0"))

# UART FC connection
uart_ok = os.path.exists('/dev/ttyTHS0')
checks.append(("FC UART", uart_ok, "/dev/ttyTHS0"))

# Performance mode
try:
    out = subprocess.check_output(['nvpmodel', '-q'], text=True)
    maxn = 'MAXN' in out
    checks.append(("Max performance mode", maxn, out.strip()))
except: checks.append(("nvpmodel", False, "not found"))

# Print results
print("\n" + "="*50)
print("  JETSON DEPLOYMENT VERIFICATION")
print("="*50)
for name, ok, detail in checks:
    status = "✓" if ok else "✗"
    print(f"  {status}  {name:30s} {detail}")
print("="*50)

all_ok = all(ok for _, ok, _ in checks)
print(f"\n  {'ALL CHECKS PASSED - READY FOR FLIGHT' if all_ok else 'ISSUES FOUND - DO NOT FLY'}")
print()
PYEOF


---

## Phase 6A Complete — Expected Performance on Jetson

| Metric | Orin Nano 8GB | Orin NX 8GB | Orin NX 16GB |
|--------|--------------|-------------|--------------|
| YOLO11n FPS (TensorRT FP16) | 80-120 | 120-160 | 160-200 |
| Inference latency | 8-12ms | 6-8ms | 5-6ms |
| Total pipeline FPS | 60-80 | 80-120 | 100-150 |
| RAM used | ~3.5GB | ~3.5GB | ~3.5GB |
| Idle power draw | ~5W | ~7W | ~10W |
| Full load power draw | ~10W | ~15W | ~20W |

TensorRT FP16 gives you the same detection accuracy as FP32
for all practical purposes — YOLO11n is robust to this quantization.
