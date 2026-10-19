# Sim validation runbook (8 pass/fail checks)

Goal: close the loop in Gazebo + ArduPilot SITL and score it with `analyze_run.py`.
Nothing here has been run on real Gazebo yet; the offline model (`tests/closed_loop.py`) is not a substitute.

## Setup
1. `python3 sim/add_camera_to_iris.py ~/ardupilot_gazebo` (adds `/drone/camera`; `--revert` undoes it).
2. `sudo apt install python3-gz-transport13 python3-gz-msgs10`.
3. Start Gazebo with `sim/worlds/tracker_world.sdf`, SITL, and `sim/move_target.py` (see `sim/launch_sim.sh`).
4. Confirm images flow: `gz topic -e -t /drone/camera | head`.
5. Run: `python3 tracker.py --config config.yaml --source gz:/drone/camera --connect 127.0.0.1:14550` (`--help` lists all flags). Set `gcs.token`.
6. Score: `python3 analyze_run.py logs/runs/<run>.csv --follow-seconds 300 --soak-seconds 1800` (exit 1 on FAIL; INCONCLUSIVE = case not exercised).

## Checks
| # | Check | How to provoke | Pass |
|---|-------|----------------|------|
| 1 | Follow | Target walks/drives 5 min | centred >=90%, range within +-20% >=80%, yaw flips <=10/10 s |
| 2 | No creep | Target stationary | no drift commands |
| 3 | Lock loss | Move target out of view | lock drops, drone hovers |
| 4 | Occlusion | Pass target behind obstacle | lock/ID recovered or cleanly dropped |
| 5 | Failsafes | Kill video; kill MAVLink; low battery | HOVER, then RTL per `safety:` |
| 6 | Pilot override | Change mode away from GUIDED | tracker releases control |
| 7 | Soak | 30 min run | no crash/leak, error counters flat |
| 8 | Logs | after every run | CSV complete, analyzer runs |

Record results per run; tune `pid:` in `config.yaml` only between runs.
