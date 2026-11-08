#!/usr/bin/env bash
# One-time setup of the Raspberry Pi 5 + AI HAT+ (Hailo-8) companion computer.
# Run ON THE PI from the repo folder, after Raspberry Pi OS 64-bit is installed:
#
#   bash scripts/setup_pi.sh                    # packages, venv, checks
#   bash scripts/setup_pi.sh --uart             # also enable the GPIO UART for the flight controller
#   bash scripts/setup_pi.sh --camera imx519    # also load a camera overlay (Arducam sensors)
#
# It prints PASS/FAIL for the Hailo chip, the camera and the Python stack at the end.
set -euo pipefail
cd "$(dirname "$0")/.."

UART=0; CAM_OVERLAY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --uart) UART=1 ;;
    --camera) CAM_OVERLAY="${2:-}"; shift ;;
    *) echo "unknown option $1"; exit 1 ;;
  esac
  shift
done

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is for the Raspberry Pi (aarch64). On the PC use scripts/setup_python_env.sh."
  exit 1
fi
CFG=/boot/firmware/config.txt

echo "== 1/5 system packages"
sudo apt-get update
sudo apt-get install -y dkms hailo-all python3-picamera2 python3-venv python3-opencv git ffmpeg

echo "== 2/5 Python venv (system site packages: picamera2 and hailo_platform come from apt)"
if [[ ! -d venv ]]; then
  python3 -m venv --system-site-packages venv
fi
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
# CPU-only PyTorch first: the default aarch64 wheel drags in ~3 GB of NVIDIA CUDA packages the Pi
# cannot use. Ultralytics needs torch to load the model; detection runs on the Hailo.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements_rpi5.txt
# pip's OpenCV would shadow the system build; the headless one avoids GUI/Qt clashes on the Pi.
pip uninstall -y opencv-python >/dev/null 2>&1 || true
pip install opencv-python-headless

if [[ $UART -eq 1 ]]; then
  echo "== 3/5 GPIO UART for the flight controller (GPIO14 TX, GPIO15 RX -> /dev/ttyAMA0)"
  sudo raspi-config nonint do_serial_cons 1     # no login console on the UART
  sudo raspi-config nonint do_serial_hw 0       # UART hardware on
  grep -q '^dtparam=uart0=on' "$CFG" || echo 'dtparam=uart0=on' | sudo tee -a "$CFG" >/dev/null
  echo "   wire Pi TX->FC RX, Pi RX->FC TX, GND->GND (3.3 V logic); FC SERIALx_PROTOCOL=2, BAUD=921"
else
  echo "== 3/5 UART skipped (add --uart when the FC is wired)"
fi

if [[ -n "$CAM_OVERLAY" ]]; then
  echo "== 4/5 camera overlay: $CAM_OVERLAY"
  sudo sed -i 's/^camera_auto_detect=1/camera_auto_detect=0/' "$CFG"
  grep -q "^dtoverlay=$CAM_OVERLAY" "$CFG" || echo "dtoverlay=$CAM_OVERLAY" | sudo tee -a "$CFG" >/dev/null
  echo "   (Raspberry Pi's own cameras are auto-detected and need no overlay)"
else
  echo "== 4/5 camera overlay skipped (auto-detect)"
fi

echo "== 5/5 checks"
ok=1
if hailortcli fw-control identify 2>/dev/null | grep -qiE 'HAILO8([^L]|$)'; then
  echo "PASS Hailo-8 detected"; hailortcli --version || true
elif hailortcli fw-control identify 2>/dev/null | grep -qi 'hailo'; then
  echo "WARN Hailo device found but not reported as HAILO8:"; hailortcli fw-control identify || true
else
  echo "FAIL no Hailo device (reboot after the first install; check the HAT ribbon)"; ok=0
fi
CAMS="$(rpicam-hello --list-cameras 2>&1 || true)"
if echo "$CAMS" | grep -qE '^[0-9]+ : '; then
  echo "PASS camera:"; echo "$CAMS" | grep -E '^[0-9]+ : '
else
  echo "FAIL no camera listed (close any other camera program; reboot if an overlay was just added; check the ribbon):"
  echo "$CAMS" | tail -3; ok=0
fi
python3 - <<'EOF' || ok=0
import ultralytics, picamera2, hailo_platform, pymavlink, cv2, numpy
print("PASS Python stack: ultralytics", ultralytics.__version__, "| numpy", numpy.__version__, "| opencv", cv2.__version__)
EOF
echo
echo "Reboot if this was the first install or the UART/camera settings changed: sudo reboot"
echo "Then: source venv/bin/activate && python3 scripts/hailo_selftest.py --source csi:0"
[[ $ok -eq 1 ]] && echo "SETUP PASS" || echo "SETUP INCOMPLETE (see FAIL lines)"
