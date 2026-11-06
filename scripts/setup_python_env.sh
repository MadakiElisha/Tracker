#!/usr/bin/env bash
# Sets up the Python side of Drone Tracker on Ubuntu 24.04.
# Run from the repo root:  bash scripts/setup_python_env.sh
set -euo pipefail
sudo apt update
sudo apt install -y python3-venv python3-pip git netcat-openbsd libgl1 libglib2.0-0
# Gazebo Python bindings (needs the Gazebo Harmonic apt repo, see docs/SETUP_UBUNTU_24_04.md)
sudo apt install -y python3-gz-transport13 python3-gz-msgs10 || \
  echo "WARN: gz python bindings not installed (add the Gazebo apt repo first). Fine for --no-fly with a video file."
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 model_download.py
python3 -m pytest tests -q
echo "OK: environment ready. Activate with: source venv/bin/activate"
