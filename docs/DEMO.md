# Exhibition demo launcher

One command starts the simulator, flies a scripted show, records it and writes a results page. It only
launches the same commands you would type by hand, plus two opt-in tracker flags (`--cmd-file`, `--record`),
so nothing you already run changes.

```bash
source ~/Tracker/venv/bin/activate          # whichever venv you use for the tracker
cd ~/Tracker
python3 demo/run_demo.py --dry-run          # prints the plan and checks files; starts nothing
python3 demo/run_demo.py                    # full show, automatic target lock
python3 demo/run_demo.py --lock manual      # you click each target (a click also works in auto mode)
python3 demo/run_demo.py --use-running-sim  # simulator already running from sim/launch_sim.sh
```

## What happens
1. Starts Gazebo + ArduPilot (`sim/launch_sim.sh ardupilot highway-fast`), waits for GPS/EKF (arming checks are never skipped).
2. Starts the tracker with the scenario's profile, recording the annotated video.
3. Takes off, then for each segment: places the target, locks it (`lock person` / `lock vehicle`), lets the drone settle, runs the target, scores it.
4. Returns home, stops everything it started, writes `logs/demo/<time>/report.html` (per-segment verdicts, speed chart, video).

Windows to arrange side by side: the Gazebo window, the tracker window, and the show-control page
(http://127.0.0.1:8090/, opened automatically; `--no-open` to skip).

## The show
`demo/scenarios/exhibition.yaml`: walker 1.5, runner 7, sprinter 10 (target hidden 1.5 s mid-run: coast and re-ID),
car 8, car 14 with lane changes, car 20 m/s. About 3.5 min of chasing plus locking, roughly 4.5 min. Edit or delete segments
freely; the launcher refuses a scenario that does not fit the 2 km road. **Only keep speeds that passed the verification
ladder** in `docs/SIM_VALIDATION.md`; the report marks each segment PASS / PARTIAL / FAIL honestly.

One mission profile covers the whole show (set by `profile:`). The slow walker segment runs with the high-speed
profile's gains; if it rocks, drop it from the scenario or run it as its own show with `profile: pedestrian`.

## Safety and cleanup
- Ctrl+C (or a crash of the tracker/sim, or lost contact with the sim) sends return-to-launch, then stops every process it started.
- Control is a local file (`logs/demo/<time>/commands.txt`), not a network port.
- If a target is not locked in time the run goes ahead and that segment is flagged, never silently passed.
- The launcher kills leftover `gz sim` / `arducopter` processes after a sim show it started itself (set `DEMO_NO_PKILL=1` to skip).

## Hardware mode (not yet flown: treat as a draft)
```bash
python3 demo/run_demo.py --mode hardware --connect /dev/ttyACM0 --monitor udpin:127.0.0.1:14551 --source 0 --duration 240
```
Runs a pre-flight checklist (link, GPS 3D fix and 8+ satellites, EKF, battery 60%+), then requires you to type `FLY`.
No scripted targets: the pilot takes off (or add `--auto-takeoff`) and you click or auto-lock a real target; after `--duration`
seconds it returns home. The pilot keeps RC override at all times. Start with `--no-fly` tracker runs on the bench first.
Needs a second MAVLink output for `--monitor`; adjust to your telemetry setup.

## Tips
- Record the Gazebo window with OBS if you want the 3D view in the video; the launcher records the tracker view itself.
- ffmpeg installed (`sudo apt install ffmpeg`) makes the video play in the browser (H.264).
- Do one full dry run the day before, and keep a recording of a good run as a fallback.
