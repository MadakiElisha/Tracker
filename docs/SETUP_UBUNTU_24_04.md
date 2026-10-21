# Setup on a fresh Ubuntu 24.04 PC

Tested pieces: the Python code and unit tests (172 tests). NOT yet verified end to end by the
author: Gazebo/SITL install steps below follow the upstream docs, and the live closed loop is
still to be run (see `SIM_VALIDATION.md`). If a step fails, the upstream docs win.

## 0. Get the code
```bash
sudo apt update && sudo apt install -y git
git clone https://github.com/MadakiElisha/Tracker.git ~/Tracker
cd ~/Tracker
git checkout production-hardening      # until it is merged into main
```

## 1. Python side (enough for tests + video-file demo, no simulator)
```bash
bash scripts/setup_python_env.sh
source venv/bin/activate
python3 -m pytest tests -q             # expect: 172 passed
```
The script creates the venv with `--system-site-packages` (required so Python can see the
Gazebo bindings installed by apt), installs `requirements.txt`, downloads the VisDrone model
(`visdrone_yolo11n.pt`) and runs the tests.

## 2. Quick check without any simulator
```bash
source venv/bin/activate
python3 tracker.py --no-fly --source /path/to/any_video.mp4
```
Keys: `L` lock, `R` release, `Q` quit. Open http://localhost:8080 for the browser view.
Set `gcs.token` in `config.yaml` before exposing the port on a network.

## 3. Gazebo Harmonic
```bash
sudo curl https://packages.osrfoundation.org/gazebo.gpg \
  --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list
sudo apt update && sudo apt install -y gz-harmonic
```
Then re-run `bash scripts/setup_python_env.sh` if the gz Python bindings were skipped earlier.

## 4. ArduPilot SITL
```bash
cd ~ && git clone --recurse-submodules https://github.com/ArduPilot/ardupilot.git
cd ardupilot
Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile                       # or log out and back in
./waf configure --board sitl && ./waf copter
```
`launch_sim.sh` uses `--no-rebuild`, so this build must exist first. If the prereq script
created `~/venv-ardupilot`, activate it in the terminal that runs `launch_sim.sh`
(`source ~/venv-ardupilot/bin/activate`) so `sim_vehicle.py`/MAVProxy are found.

## 5. ArduPilot Gazebo plugin
```bash
sudo apt install -y libgz-sim8-dev rapidjson-dev libopencv-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl
cd ~ && git clone https://github.com/ArduPilot/ardupilot_gazebo.git
cd ardupilot_gazebo && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo && make -j4
```
The launcher expects it at `~/ardupilot_gazebo`.

## 6. Give the drone a camera (one time)
```bash
cd ~/Tracker
python3 sim/add_camera_to_iris.py ~/ardupilot_gazebo      # undo with --revert
```

## 6b. Download the realistic sim models (one time, needs internet)
```bash
cd ~/Tracker && ./sim/fetch_models.sh      # ~100 MB of CC0 models from Gazebo Fuel, cached in ~/.gz/fuel
```
The world uses real-looking people, cars, a bus, trees and cones, because the VisDrone model does not
detect plain boxes. Moving actors: `moving_target` (person), `moving_person2`, `vehicle_target` (pickup), `moving_suv`.

## 7. Run the full sim (3 terminals)
```bash
# Terminal 1 - Gazebo + SITL (ends at the MAVProxy prompt)
cd ~/Tracker/sim && ./launch_sim.sh ardupilot

# Terminal 2 - moving targets
cd ~/Tracker && source venv/bin/activate && python3 sim/move_target.py

# Terminal 3 - tracker on the drone camera
cd ~/Tracker && source venv/bin/activate
gz topic -e -t /drone/camera | head -c 300     # sanity check: image data appears
python3 tracker.py --connect 127.0.0.1:14550 --source gz:/drone/camera
```
Press `T` to take off, click a box (or `L`) to lock, `H` for RTL, `Q` to quit.
Flags: `--no-fly` (perception only), `--no-display` (headless), `--config other.yaml`.

## 8. Score a run
```bash
python3 analyze_run.py logs/runs/<run>.csv --follow-seconds 300 --soak-seconds 1800
```
See `SIM_VALIDATION.md` for the 8 checks.

## Troubleshooting
- `ModuleNotFoundError: gz` -> venv was made without `--system-site-packages`; recreate it.
- `Model file not found: visdrone_yolo11n.pt` -> `python3 model_download.py`.
- No frames from `gz:` -> camera patch not applied, or Gazebo restarted before the patch.
- SITL never opens port 5760 -> ArduPilot not built, or its venv not active.
- Browser view not reachable from another PC -> firewall; and set `gcs.token`.
