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

| Profile | Cap | Altitude | Targets |
|---|---|---|---|
| pedestrian | 4 m/s | 6 m | walking, jogging |
| runner | 10 m/s | 6 m | sprinters |
| vehicle | 16 m/s | 10 m | cars, vans, buses |
| motorcycle | 20 m/s | 8 m | motorbikes, cyclists |

Offline model (`python3 tests/usecases.py [--gimbal off]`, also run by pytest):
- **Fixed camera (no gimbal): 10 of 12 pass.** Fails: stop-and-go traffic (target visible 96% of the
  time, just under the 98% bar: hard stops flip the lean) and the 28 m/s highway car.
- **With a gimbal: 11 of 12 pass.** Only the highway car fails: it exceeds the modelled 20 m/s airframe.

**What made the fixed camera work.** The drone leans 30-45 deg at speed and the camera leans with it.
The controller now (a) reads tilt from MAVLink ATTITUDE and corrects the vertical image error, and
(b) uses that corrected error in the forward-speed gate. Before (b), each lean closed the gate, which
cut speed, which flipped the lean: a 4-16 m/s oscillation that threw the target out of frame
(fixed camera passed 2 of 12; now 10 of 12).

**Limits.** This is a model, not Gazebo and not a flight. Not yet validated in Gazebo or on hardware.
A 10-15 inch frame leans harder than the model's airframe, so expect less margin; a gimbal, or a
camera mounted with extra up-tilt, is the next step if real flights lose the target. No obstacle avoidance.

**Target speeds in Gazebo.** `python3 sim/move_target.py --person-speed 1.4` (walk, default),
`--person-speed 3` (jog), `--vehicle-speed 8`. Speeds are wall-clock exact. The old fixed circle
moved the person at 3.1 m/s, a fast jog, not a walk.
