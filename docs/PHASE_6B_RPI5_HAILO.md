# Phase 6B — Raspberry Pi 5 + Hailo-8L AI Hat Deployment Guide
# Drone Tracker — Full Hardware Deployment

## Hardware Required
- Raspberry Pi 5 (8GB recommended, 4GB minimum)
- Raspberry Pi AI Hat+ (Hailo-8L — 26 TOPS)  OR  Hailo-8 (26 TOPS, same chip)
- MicroSD card (64GB+ A2 rated) or NVMe SSD via M.2 HAT
- Raspberry Pi Camera Module 3 (or Arducam IMX477 Global Shutter)
- USB-to-UART adapter (for FC bench testing) or direct UART via GPIO
- 5V/5A USB-C power supply (official RPi 5 PSU) or regulated 5V from PDB

## Why Hailo-8L Over Hailo-8
Both chips have 26 TOPS. Hailo-8L is physically smaller and lower power,
designed exactly for edge/embedded scenarios like ours. Performance is
identical for YOLO11n inference. Hailo-8 is the same die with higher
sustained power budget — useful for larger models but unnecessary here.

---

## STEP 1 — OS Setup

### 1.1 Flash Raspberry Pi OS
Use Raspberry Pi Imager on your dev PC:
  Download: https://www.raspberrypi.com/software/

Settings to configure before writing:
  OS:        Raspberry Pi OS (64-bit) Bookworm — FULL version
  Hostname:  dronetracker
  Username:  drone
  Password:  <your choice>
  SSH:       Enable
  WiFi:      Set your network (for initial setup only)

Flash to MicroSD, insert into RPi 5, power on.

### 1.2 First boot via SSH
    ssh drone@dronetracker.local

### 1.3 Update system
    sudo apt update && sudo apt full-upgrade -y
    sudo reboot

### 1.4 Enable required interfaces
    sudo raspi-config
    # Interface Options:
    #   → Camera: Enable
    #   → Serial Port: Enable (login shell: NO, hardware: YES)
    #   → I2C: Enable
    #   → SPI: Enable
    # Finish → Reboot

### 1.5 Verify 64-bit OS (required for Hailo)
    uname -m
    # Must print: aarch64
    # If it prints armv7l, you flashed the 32-bit version — reflash


---

## STEP 2 — Hailo-8L AI Hat Installation

### 2.1 Physical installation
The AI Hat+ connects to the RPi 5 via the M.2 slot on the underside
of the board. The HAT stacks on top via GPIO passthrough.

Installation order:
1. Attach the M.2 spacers to RPi 5
2. Insert AI Hat+ into M.2 slot at 45° angle, press flat, secure screw
3. Stack HAT on top of RPi 5 via GPIO header
4. Connect the FPC ribbon cable between HAT and RPi 5 (critical for PCIe)
5. Connect ribbon cable for the camera to the HAT's camera port
   (NOT directly to RPi 5 — the HAT passes camera through itself)

### 2.2 Install Hailo software stack
    # Add Hailo repository
    sudo apt install -y hailo-all

    # This installs:
    #   - hailort (runtime library)
    #   - hailo kernel driver (pcie-hailo8)
    #   - hailo-tappas (GStreamer plugins)
    #   - python3-hailo (Python bindings)

    sudo reboot

### 2.3 Verify Hailo device is detected
    hailortcli scan
    # Expected output:
    # Hailo Devices:
    # [-] Device: 0000:01:00.0

    # Check firmware version
    hailortcli fw-control identify
    # Expected: Device information + firmware version

    # Verify Python bindings
    python3 -c "import hailo; print('Hailo OK:', hailo.__version__)"

### 2.4 Run Hailo system check
    hailortcli scan
    # Must show a device. If empty:
    #   1. Check FPC ribbon cable is fully seated
    #   2. Check M.2 screw is tight
    #   3. Run: dmesg | grep -i hailo
    #      Look for PCIe enumeration messages


---

## STEP 3 — Install Project Dependencies

### 3.1 System packages
    sudo apt install -y \
        python3-pip \
        python3-venv \
        python3-dev \
        python3-opencv \
        git \
        cmake \
        build-essential \
        v4l-utils \
        libcamera-apps \
        python3-libcamera \
        python3-picamera2 \
        libopenblas-dev \
        libatlas-base-dev

### 3.2 Create virtual environment
    cd ~
    git clone <your drone_tracker repo>
    cd drone_tracker

    # --system-site-packages inherits picamera2, libcamera, hailo bindings
    python3 -m venv venv --system-site-packages
    source venv/bin/activate

### 3.3 Install Python packages
    pip install --upgrade pip

    pip install \
        pymavlink \
        pyserial \
        flask \
        supervision \
        scipy

    # Ultralytics — standard install, inference via HailoRT not PyTorch
    pip install ultralytics

    # BoxMOT for ByteTrack
    pip install boxmot

    # Install numpy compatible with RPi 5
    pip install "numpy>=1.24,<2.0"

### 3.4 Verify hailo is importable inside venv
    python3 -c "
    import hailo
    import picamera2
    print('Hailo:', hailo.__version__)
    print('Picamera2: OK')
    print('All RPi dependencies verified.')
    "


---

## STEP 4 — YOLO11n to Hailo .hef Export

The .hef (Hailo Executable Format) is the compiled model that runs
on the Hailo-8L NPU. Unlike TensorRT, .hef files ARE portable between
Hailo-8L devices of the same generation — you can export on your dev
PC (with Hailo Dataflow Compiler installed) and deploy to the RPi.

### 4.1 Option A — Export on dev PC (recommended)
Install Hailo Dataflow Compiler on Ubuntu 22.04 (dev machine):

    # Register and download from Hailo Developer Zone:
    # https://developer.hailo.ai/developer-zone/

    # Download: hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl
    pip install hailo_dataflow_compiler-<version>-py3-none-linux_x86_64.whl

    # Export YOLO11n to HEF
    python3 << 'PYEOF'
from ultralytics import YOLO

print("Loading YOLO11n...")
model = YOLO("yolo11n.pt")

print("Exporting to Hailo HEF format...")
print("This takes 10-30 minutes on first run.")
model.export(
    format  = "hailo",
    device  = "hailo",
    imgsz   = 640,
    half    = True,    # INT8 quantization for Hailo
    batch   = 1,
)
print("Export complete: yolo11n.hef")
PYEOF

    # Copy the .hef file to RPi 5:
    scp yolo11n.hef drone@dronetracker.local:~/drone_tracker/

### 4.2 Option B — Export on RPi 5 directly
Hailo's Python SDK on the RPi can also run the export, but it is
significantly slower. Expect 30-90 minutes.

    # On the RPi 5:
    source ~/drone_tracker/venv/bin/activate
    python3 -c "
    from ultralytics import YOLO
    model = YOLO('yolo11n.pt')
    model.export(format='hailo', imgsz=640, half=True, batch=1)
    "

### 4.3 Verify the HEF file on RPi 5
    # Check file exists and has content
    ls -lh ~/drone_tracker/yolo11n.hef
    # Should be ~5-15MB

    # Run Hailo's own inspection tool
    hailortcli parse-hef yolo11n.hef
    # Shows: network name, input/output shapes, ops count


---

## STEP 5 — HailoRT Inference Integration

Replace the standard Ultralytics model.track() call with a
HailoRT-native inference pipeline. Ultralytics has built-in Hailo
support in version 8.3+, so the API stays the same.

### 5.1 Test HailoRT inference speed
    python3 << 'PYEOF'
from ultralytics import YOLO
import numpy as np
import time

print("Loading Hailo HEF model...")
model = YOLO("yolo11n.hef", task="detect")

# Warm up
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
print("Warming up (3 passes)...")
for _ in range(3):
    model(dummy, verbose=False)

# Benchmark
print("Benchmarking 100 frames...")
t = time.time()
for _ in range(100):
    model(dummy, verbose=False)
elapsed = time.time() - t

print(f"100 frames in {elapsed:.2f}s")
print(f"Throughput: {100/elapsed:.1f} FPS")
print(f"Latency:    {elapsed/100*1000:.1f}ms per frame")
PYEOF
    # Expected: 30-45 FPS, 22-33ms latency

### 5.2 Update YOLO_MODEL in tracker
    # In tracker2bi.py:
    YOLO_MODEL = "yolo11n.hef"
    # No other code changes needed — Ultralytics handles HailoRT transparently


---

## STEP 6 — Camera Integration (RPi Camera Module 3)

RPi Camera Module 3 uses Sony IMX708 — this has a rolling shutter.
For slow-moving targets it's fine. For fast FPV at speed, consider
the Arducam IMX477 Global Shutter (connects to same CSI port).

### 6.1 Connect camera
- RPi Camera Module 3 → CSI port on AI Hat+ (NOT directly on RPi 5)
- The AI Hat routes the camera through its own CSI bridge
- Use the short ribbon cable that ships with the AI Hat+

### 6.2 Verify camera detection
    libcamera-hello --list-cameras
    # Should show: Available cameras + sensor info

    # Quick capture test
    libcamera-still -o /tmp/test.jpg
    # Should save a JPEG without errors

### 6.3 Access camera in OpenCV via Picamera2

Standard cv2.VideoCapture(0) may not work with the HAT bridge.
Use picamera2 directly — it handles the HAT camera routing:

    PICAM2_AVAILABLE = False
    try:
        from picamera2 import Picamera2
        PICAM2_AVAILABLE = True
    except ImportError:
        pass

    def open_camera(frame_w=640, frame_h=480):
        """Open camera — uses Picamera2 on RPi, cv2 elsewhere."""
        if PICAM2_AVAILABLE:
            cam = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (frame_w, frame_h), "format": "BGR888"},
                controls={"FrameRate": 60}
            )
            cam.configure(config)
            cam.start()
            print("[CAMERA] Picamera2 initialized.")
            return cam, 'picamera2'
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
            print("[CAMERA] OpenCV VideoCapture initialized.")
            return cap, 'opencv'

    def read_frame(cam, cam_type):
        """Read one frame — handles both camera backends."""
        if cam_type == 'picamera2':
            frame = cam.capture_array()
            return True, frame
        else:
            return cam.read()

### 6.4 Full camera pipeline test
    python3 << 'PYEOF'
from picamera2 import Picamera2
import cv2, time

cam = Picamera2()
config = cam.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"},
    controls={"FrameRate": 60}
)
cam.configure(config)
cam.start()
print("Camera open. Running for 5s to measure FPS...")

count = 0
t_start = time.time()
while time.time() - t_start < 5:
    frame = cam.capture_array()
    count += 1

fps = count / 5
print(f"Camera FPS: {fps:.1f}")
print("Camera integration OK." if fps > 20 else "WARNING: FPS too low.")
cam.stop()
PYEOF


---

## STEP 7 — MAVLink Over UART to Flight Controller

### 7.1 Wiring — RPi 5 GPIO UART
RPi 5 GPIO header UART0 (default):
  Pin 8  (GPIO14 / TXD) ────► FC UART RX
  Pin 10 (GPIO15 / RXD) ◄──── FC UART TX
  Pin 6  (GND)          ────── FC GND
  Do NOT connect 5V between RPi and FC.

UART device: /dev/ttyAMA0  (or /dev/serial0 symlink)

ArduPilot FC settings:
  SERIAL1_BAUD     = 921  (921600)
  SERIAL1_PROTOCOL = 2    (MAVLink 2)
  BRD_SER1_RTSCTS  = 0    (no flow control)

### 7.2 Disable serial console (frees UART for MAVLink)
    sudo raspi-config
    # Interface Options → Serial Port
    #   Login shell: NO
    #   Serial hardware: YES
    # Reboot

    # Verify UART is free
    ls -la /dev/serial*
    # /dev/serial0 → ttyAMA0  (this is what we use)

### 7.3 Test UART MAVLink connection
    # Update fc_interface5.py connection string:
    # Change: 'udpin:127.0.0.1:14550'
    # To:     '/dev/serial0'  with baud=921600

    python3 fc_interface5.py --connect /dev/serial0
    # Expected: heartbeat from FC within 5s


---

## STEP 8 — Tracker Startup Script With Hardware Detection

This script auto-detects which hardware it's running on and
configures everything correctly before launching the tracker.

    cat > ~/drone_tracker/start.sh << 'SCRIPT'
    #!/bin/bash
    # Drone Tracker startup script
    # Auto-detects Jetson vs RPi vs dev PC

    cd "$(dirname "$0")"
    source venv/bin/activate

    # Detect platform
    if [ -f /usr/bin/nvpmodel ]; then
        PLATFORM="jetson"
        FC_CONNECT="/dev/ttyTHS0"
        echo "[START] Platform: Jetson"
    elif [ -f /proc/device-tree/model ] && grep -q "Raspberry Pi 5" /proc/device-tree/model; then
        PLATFORM="rpi5"
        FC_CONNECT="/dev/serial0"
        echo "[START] Platform: Raspberry Pi 5"
    else
        PLATFORM="dev"
        FC_CONNECT="127.0.0.1:14550"
        echo "[START] Platform: Development PC (SITL)"
    fi

    echo "[START] FC connection: $FC_CONNECT"

    python3 tracker2bi.py \
        --fc ardupilot \
        --connect "$FC_CONNECT" \
        ${1:---no-display}

    SCRIPT
    chmod +x ~/drone_tracker/start.sh

### 8.1 Autostart on boot
    sudo tee /etc/systemd/system/drone-tracker.service << 'EOF'
    [Unit]
    Description=Drone Tracker Autonomous Follow System
    After=network.target
    Wants=network.target

    [Service]
    Type=simple
    User=drone
    WorkingDirectory=/home/drone/drone_tracker
    Environment=PYTHONUNBUFFERED=1
    ExecStart=/home/drone/drone_tracker/start.sh --no-display
    Restart=on-failure
    RestartSec=5

    [Install]
    WantedBy=multi-user.target
    EOF

    sudo systemctl daemon-reload
    sudo systemctl enable drone-tracker.service

### 8.2 Control commands
    sudo systemctl start drone-tracker
    sudo systemctl stop drone-tracker
    sudo systemctl status drone-tracker
    journalctl -u drone-tracker -f          # live logs


---

## STEP 9 — Power Management and Weight Budget

### 9.1 Power consumption
    RPi 5 idle:              ~3W
    RPi 5 full load:         ~7W
    Hailo-8L inference:      ~2-3W additional
    Camera (IMX708):         ~0.5W
    Total system max:        ~10-11W

    At 5V this is ~2.2A peak draw.
    Use a BEC rated for 3A minimum from your drone PDB.
    Add a low-ESR capacitor (1000µF 16V) across the 5V rail
    to absorb motor noise spikes.

### 9.2 Weight
    RPi 5 board:             ~50g
    Hailo-8L AI Hat+:        ~30g
    Camera module:           ~5g
    Case/mounting:           ~20g
    Wiring:                  ~10g
    Total:                   ~115g

    For a 5" FPV quad (typically 600-800g AUW), this is significant.
    Mount as close to CG as possible.
    For a 7" or 10" build, weight is much less of a concern.

### 9.3 Thermal considerations
RPi 5 throttles at 85°C. Add a heatsink and small 5V fan.
The AI Hat generates additional heat — ensure airflow over both boards.

    # Monitor temperature
    vcgencmd measure_temp
    # Should stay below 70°C under sustained inference load

    # If throttling:
    vcgencmd get_throttled
    # 0x0 = fine, any other value = thermal/voltage issues


---

## STEP 10 — Full System Verification Checklist

    python3 << 'PYEOF'
import sys, os, subprocess

checks = []

# Hailo device
hailo_ok = os.path.exists('/dev/hailo0')
checks.append(("Hailo-8L device", hailo_ok, "/dev/hailo0"))

# HEF model
hef_ok = os.path.exists('yolo11n.hef')
checks.append(("YOLO11n HEF model", hef_ok, "yolo11n.hef"))

# Picamera2
try:
    import picamera2
    checks.append(("Picamera2", True, picamera2.__version__))
except ImportError:
    checks.append(("Picamera2", False, "not installed"))

# UART
uart_ok = os.path.exists('/dev/serial0')
checks.append(("FC UART", uart_ok, "/dev/serial0"))

# Temperature
try:
    out = subprocess.check_output(['vcgencmd', 'measure_temp'], text=True).strip()
    temp = float(out.replace("temp=", "").replace("'C", ""))
    ok = temp < 75
    checks.append(("CPU Temperature", ok, out))
except:
    checks.append(("vcgencmd", False, "not found"))

# Hailo inference speed
if hailo_ok and hef_ok:
    try:
        from ultralytics import YOLO
        import numpy as np, time
        model = YOLO("yolo11n.hef", task="detect")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3): model(dummy, verbose=False)
        t = time.time()
        for _ in range(30): model(dummy, verbose=False)
        fps = 30 / (time.time() - t)
        ok = fps > 20
        checks.append(("Hailo inference FPS", ok, f"{fps:.1f} FPS"))
    except Exception as e:
        checks.append(("Hailo inference", False, str(e)))

# Print results
print("\n" + "="*50)
print("  RPi 5 + HAILO DEPLOYMENT VERIFICATION")
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

## Phase 6B Complete — Expected Performance on RPi 5 + Hailo-8L

| Metric | RPi 5 (CPU) | RPi 5 + Hailo-8L | RPi 5 + Hailo-8 |
|--------|-------------|------------------|-----------------|
| YOLO11n FPS | 8-12 | 30-45 | 30-45 |
| Inference latency | 85-120ms | 22-33ms | 22-33ms |
| Total pipeline FPS | 6-10 | 25-40 | 25-40 |
| RAM used | ~1.5GB | ~1.5GB | ~1.5GB |
| Power draw (full) | ~7W | ~10W | ~10W |
| Weight | ~50g | ~80g | ~80g |

YOLO11n is well within Hailo-8L's capabilities.
If you later want YOLO11s (larger, more accurate), it still runs
at ~20-25 FPS on Hailo-8L — still flight-capable.

The RPi 5 + Hailo-8L stack is significantly lighter and cheaper
than the Jetson Orin Nano while delivering comparable performance
for this specific workload. The Jetson wins at >30% larger models
and multi-camera setups.
