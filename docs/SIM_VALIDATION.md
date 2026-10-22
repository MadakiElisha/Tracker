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
| 2 | No creep | Lock a STATIONARY target (parked car / standing person), hold ~60 s | no drift commands; run `analyze_run.py FILE --locks` to find the window, then `--rest-window START END` |
| 3 | Lock loss | Move target out of view | lock drops, drone hovers |
| 4 | Occlusion | Pass target behind obstacle | lock/ID recovered or cleanly dropped |
| 5 | Failsafes | Kill video; kill MAVLink; low battery | HOVER, then RTL per `safety:` |
| 6 | Pilot override | Change mode away from GUIDED | tracker releases control |
| 7 | Soak | 30 min run | no crash/leak, error counters flat |
| 8 | Logs | after every run | CSV complete, analyzer runs |

Record results per run; tune `pid:` in `config.yaml` only between runs.

## Speed use cases and mission profiles

`config.yaml` has a `profiles:` section; select one with `python3 tracker.py --profile vehicle ...`.

| Profile | Cap | Altitude | Camera | Targets |
|---|---|---|---|---|
| pedestrian | 4 m/s | 6 m | fixed OK | walking, jogging |
| runner | 10 m/s | 6 m | **gimbal** | sprinters |
| vehicle | 16 m/s | 10 m | **gimbal** | cars, vans, buses |
| motorcycle | 20 m/s | 8 m | **gimbal** | motorbikes, cyclists |

Offline model matrix (`python3 tests/usecases.py`, also run by pytest): 11 of 12 pass. The
28 m/s highway case fails by design: it exceeds the modelled 20 m/s airframe.

**Limits.** This is a model, not Gazebo and not a flight. Above ~4 m/s a fixed camera loses the
target as the drone leans (30-45 deg at speed), so those profiles assume a stabilised gimbal.
Tilt comes from MAVLink ATTITUDE, altitude and ground speed from GLOBAL_POSITION_INT. Not yet
validated in Gazebo; `sim/move_target.py` still moves actors at 10 Hz, which is too coarse for
10+ m/s targets. There is no obstacle avoidance.

**Target speeds in Gazebo.** `python3 sim/move_target.py --person-speed 1.4` (walk, default),
`--person-speed 3` (jog), `--vehicle-speed 8`. Speeds are wall-clock exact. The old fixed circle
moved the person at 3.1 m/s, a fast jog, not a walk.
