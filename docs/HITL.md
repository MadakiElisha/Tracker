# Hardware-in-the-loop (HITL-A): the real tracker on the Pi, flying the simulated drone

The PC runs Gazebo + ArduPilot SITL and relays the drone camera; the **Pi 5 + Hailo-8 runs the
real tracker** (detection, ByteTrack, lock, re-ID, PID, safety) and commands SITL over MAVLink.
This measures what the Gazebo-on-PC runs cannot: the Pi's real frame rate, the Hailo model's
detections, and the whole loop's latency, before anything flies.

```
 PC (10.42.0.1)                                   Pi 5 + Hailo-8 (10.42.0.2)
 Gazebo ── camera ── sim/frame_relay.py ──TCP 5601──▶ tracker.py --source net:10.42.0.1:5601
 ArduPilot SITL ◀───────────── MAVLink UDP 14550 ──── tracker.py --connect 0.0.0.0:14550
```

## Prerequisites
- Direct Ethernet cable, fixed addresses (Pi setup guide, step 8): PC `10.42.0.1`, Pi `10.42.0.2`, `ping` works.
- The Hailo model folder(s) on the Pi, and `scripts/hailo_selftest.py` passing (≥ 19 fps).
- Both machines on the same commit of this repo.
- Optional, for meaningful latency numbers: the same time source on both
  (`sudo apt install chrony` on both; on the Pi add `server 10.42.0.1 iburst` to `/etc/chrony/chrony.conf`).

## Run
**PC, terminal 1:** simulation with MAVLink also sent to the Pi
```bash
cd ~/Tracker
HITL_OUT=10.42.0.2 ./sim/launch_sim.sh ardupilot            # or: ... ardupilot highway
```
**PC, terminal 2:** camera relay
```bash
python3 sim/frame_relay.py            # prints captured/sent fps, KB per frame, drops every 5 s
```
**Pi:** the tracker (Gazebo camera is 90 deg: no --camera preset; the leash uses the sim value automatically)
```bash
cd ~/Tracker && source venv/bin/activate
python3 tracker.py --source net:10.42.0.1:5601 --connect 0.0.0.0:14550 --no-display \
                   --profile pedestrian --model yolo11n_hailo_model
```
Open `http://10.42.0.2:8080` on the PC for the tracker's view; take off and lock as in the Gazebo runs
(or `python3 demo/run_demo.py ...` scenarios, which use the same keys/GCS).

## What to check (send the run CSV from `logs/runs/` and `logs/tracker.log`)
| Check | Pass |
|---|---|
| Relay | ~30 fps sent, 0 dropped, < 60 KB/frame |
| Tracker fps on the Pi (run CSV `fps`) | median ≥ 19 (what the profiles were tuned at) |
| Detector time (`det_ms`) | median ≤ 40 ms on the Hailo |
| Walker / jogger scenario | `analyze_run.py` PASS, as on the PC runs |
| Failsafes | unplug the Ethernet: hover within 0.5 s, RTL after 5 s (video loss); stop SITL output: hover/release (link loss) |
| Pi health | no `PI HEALTH` warnings in `tracker.log` over a 10-minute run |

`python3 analyze_run.py logs/runs/<file>.csv` scores the run exactly as for PC runs.

## Not covered by HITL-A
The real camera, real flight controller, radio and power: that is HITL-B, the bench test with the
real FC (props off), which follows this.
