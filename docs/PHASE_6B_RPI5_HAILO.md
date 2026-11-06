# Raspberry Pi 5 + AI HAT+ 26 TOPS (Hailo-8): setup and model deployment

Hardware this guide is written for: Raspberry Pi 5 8 GB, Raspberry Pi AI HAT+ **26 TOPS (Hailo-8)**,
a CSI ribbon camera (IMX296 M12 96° for flight; IMX519 as the interim camera), ArduPilot flight
controller on the Pi's GPIO UART. The previous version of this guide is in `legacy/`.

The detector is **YOLO11n trained on VisDrone** (`dronefreak/visdrone-yolov11n`). On the Pi it runs on the
Hailo chip as a compiled HEF; everything after detection (ByteTrack, lock, re-ID, PID, safety) is the same
code that runs in Gazebo.

## Order matters

The Hailo compiler (DFC) on the PC must match the HailoRT runtime on the Pi, so set up the Pi first.

| Step | Where | What |
|---|---|---|
| 1 | Pi | Raspberry Pi OS 64-bit (Trixie) with Raspberry Pi Imager: hostname, user, SSH on |
| 2 | Pi | `git clone` the repo, `bash scripts/setup_pi.sh`, reboot. Note `hailortcli --version` |
| 3 | PC | Download the **DFC v3.x, Linux x86_64** wheel matching that HailoRT version (Hailo Developer Zone, free account) |
| 4 | PC | `bash scripts/setup_hailo_dfc.sh ~/Downloads/hailo_dataflow_compiler-*.whl` |
| 5 | PC | `python3 model_download.py` (if `visdrone_yolo11n.pt` is not there yet) |
| 6 | PC | `source venv-hailo/bin/activate && python3 deployment/export_hailo.py` (20–60 min) |
| 7 | PC → Pi | `scp -r visdrone_yolo11n_hailo_model <user>@<pi>:~/Tracker/` |
| 8 | Pi | `python3 scripts/hailo_selftest.py --source csi:0 --camera imx519_78` → PASS at ≥ 19 fps |

## 1–2. The Pi

```bash
sudo apt update && sudo apt full-upgrade -y
git clone https://github.com/MadakiElisha/Tracker.git && cd Tracker
bash scripts/setup_pi.sh                 # packages, venv, PASS/FAIL checks
sudo reboot
```
`setup_pi.sh` installs `hailo-all` (driver, firmware, HailoRT, Python bindings) and `python3-picamera2`,
then makes `venv/` with `--system-site-packages` so the venv sees both. Expect `PASS Hailo-8 detected`;
`HAILO8L` there means the 13 TOPS HAT and the model must be exported with `--arch hailo8l`.

Cameras: Raspberry Pi's own cameras are auto-detected. Arducam sensors may need an overlay:
`bash scripts/setup_pi.sh --camera imx519` (sets `camera_auto_detect=0` and `dtoverlay=imx519`), reboot,
then `rpicam-hello --list-cameras` must list the sensor.

## 3–6. Compile VisDrone for the Hailo-8 (PC)

The DFC wheel supports specific Python versions; `setup_hailo_dfc.sh` reads the wheel's tag and builds a
separate `venv-hailo/` with a matching interpreter, so the tracker's own venv is untouched.

`export_hailo.py` runs Ultralytics' Hailo export: `.pt → ONNX → Hailo → INT8 → HEF`, target `hailo8`.
INT8 calibration uses real VisDrone validation images (`data=VisDrone.yaml`, about 2 GB, downloaded once);
pass `--data` to use a local copy. The on-chip NMS score floor is 0.10 on purpose: ByteTrack uses
low-score boxes, and `tracker.py` applies `model.conf_threshold` itself.

Output: `visdrone_yolo11n_hailo_model/` (the `.hef` plus `metadata.yaml`). Copy the **whole folder**.

## 8. Run on the Pi

```bash
source venv/bin/activate
python3 scripts/hailo_selftest.py --source csi:0 --camera imx519_78      # speed + detections, no FC
python3 tracker.py --no-fly --source csi:0 --camera imx519_78 --profile pedestrian
```
With `model.backend: auto` the tracker uses the Hailo folder whenever `/dev/hailo0` exists and falls back to
the `.pt` file otherwise (the log line `Model loaded: ... (backend: hailo)` confirms which). Force either with
`--model visdrone_yolo11n_hailo_model` or `--model visdrone_yolo11n.pt`.

`hailo_selftest.py --model visdrone_yolo11n.pt` runs the same test on the Pi's CPU for comparison.

## Camera notes

`--source csi:N` uses Picamera2 (`csi_camera.py`): frames at 640×480 in BGR, focus **locked** on autofocus
sensors (`camera.csi.lens_position`, 0.0 = infinity), optional fixed short shutter
(`camera.csi.max_exposure_us`, default 3000 µs) with automatic gain, and always the newest frame.

`--camera NAME` sets the field of view the controller uses. It must match the lens:

| Preset | Sensor / lens | hfov at 640×480 | Use |
|---|---|---|---|
| `imx296_m12_96` | IMX296 global shutter, M12 96° | 83.3° (if 96° is diagonal) | flight camera |
| `imx519_78` | IMX519 rolling shutter, AF locked | 66.3° (if 78.5° is diagonal) | interim flight camera |
| `imx477_12mm` | IMX477 rolling shutter, 12 mm | 29.4° | bench only (too narrow to follow) |
| `gazebo` | sim camera | 90° | Gazebo |

Confirm the vendor FOV figures (diagonal or horizontal) before flying.

## Flight controller UART

```bash
bash scripts/setup_pi.sh --uart      # disables the serial console, enables /dev/ttyAMA0 (GPIO14 TX, GPIO15 RX)
```
Wire Pi TX → FC RX, Pi RX → FC TX, GND → GND. On the FC: `SERIALx_PROTOCOL 2` (MAVLink2),
`SERIALx_BAUD 921`. Run the tracker with `--connect /dev/ttyAMA0`. Power the Pi from its own 5 V supply
rated for at least 5 A; a brown-out mid-flight drops the companion computer.

## Not yet verified on the hardware

Hailo inference through Ultralytics, the CSI source on a real sensor and the frame rate on the Pi are
covered by tests on the PC only. `hailo_selftest.py` is the first check to run on the Pi.
