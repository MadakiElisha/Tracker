# ArduCopter setup: 13" MVP quad (DAKEFPV H743, Bluejay, ELRS, Pi 5 tracker)

Goal: an airframe that flies well **on its own** (pilot, Loiter) before the tracker ever commands it.
Do the steps in order; each ends with a check. Props **off** until step 9.

Parameter file: `params/mvp_13in_dakeh743.param`. Board reference: [ArduPilot DAKEFPV H743 / H743 Pro](https://ardupilot.org/copter/docs/common-dakefpvh743pro.html).

| # | Step | Check |
|---|---|---|
| 1 | **Firmware.** Latest stable ArduCopter. Target `DAKEFPVH743Pro` or `DAKEFPVH743` (they differ: check the board). First flash by DFU (hold the boot button while plugging USB, flash `*_with_bl.hex`); later updates via Mission Planner. | Mission Planner connects; HUD moves with the board |
| 2 | **ESC firmware.** Flash Bluejay (esc-configurator.com, through the FC's passthrough) if the ESC still runs BLHeli_S. Bluejay is what makes bidirectional DShot (RPM) work. | All 4 ESCs show Bluejay |
| 3 | **Mandatory setup:** frame Quad X, accelerometer calibration (6 positions), compass calibration (GPS mast fitted, away from battery and ESC leads), radio calibration. | No PreArm "compass" or "accel" messages |
| 4 | **Radio (Jumper T15, ELRS 4.0).** Receiver on the FC's UART5 (RX5/TX5). Map switches: CH5 arm (2-pos), CH6 flight mode (3-pos), CH7 lock (momentary or 2-pos), CH8 release, CH9 class filter (3-pos), CH10 RTL. In the ELRS Lua keep Switch Mode at the default (Hybrid): CH5 is then the 2-position arm channel and CH6-CH12 carry 3/6-position switches. | Radio calibration page: every switch moves its channel 1000 / 1500 / 2000 |
| 5 | **Motor order and direction.** Motor Test (props OFF), 5 %. ArduPilot X order: 1 front-right CCW, 2 rear-left CCW, 3 front-left CW, 4 rear-right CW. If the stack follows Betaflight order, set `FRAME_TYPE 12` rather than re-soldering. Reverse direction in Bluejay if needed. | Each test letter spins the right motor the right way |
| 6 | **Initial tune parameters.** Mission Planner > Setup > Mandatory Hardware > *Initial Tune Parameters*: 13 in props, 6 cells, 4.2 / 3.3 V. Write them. (Gyro filter, rate filters, accel limits and thrust expo for this prop size.) | Parameters written |
| 7 | **Load the MVP file.** Config > Full Parameter List > *Load from file* `mvp_13in_dakeh743.param` > Write > reboot. | Compare shows the values; reboot clean |
| 8 | **Bench checks (props off, battery in).** (a) Arm at `MOT_SPIN_ARM`: all 4 spin evenly; adjust `MOT_SPIN_ARM` / `MIN`. (b) ESC RPM appears in the status tab (`esc1_rpm`...) once armed: bidirectional DShot works. (c) Battery voltage matches a multimeter within 0.1 V (`BATT_VOLT_MULT`). (d) Radio off → failsafe message, motors stop (on the bench, disarmed in failsafe is fine). (e) Flight-mode switch shows AltHold / Loiter / Guided. (f) Fence enabled, GPS 3D fix with 10+ satellites outdoors. | All six pass |
| 9 | **First hover (props on, open field).** Take off in AltHold, then Loiter, 3-5 m, 1-2 minutes, gentle. Land. | Holds position; no oscillation; motors not hot |
| 10 | **Read the first log** (send it to Claude): hover throttle (`MOT_THST_HOVER` learned), vibration (`VIBE` < 30 m/s², clipping 0), current draw vs. capacity used, ESC RPM → notch frequency; FFT of the batch-sampled gyro. Then set `INS_LOG_BAT_MASK 0`. | Notch centred on the motor noise; vibration OK |
| 11 | **AutoTune** in AltHold (`AUTOTUNE_AXES 7`, aggressiveness default), in calm wind, one axis at a time if the battery is short; save by landing and disarming in AutoTune. | Crisp in Loiter; no wobble on fast stops |
| 12 | **Battery calibration.** After a flight, compare mAh used (log) with the charger's mAh put back; correct `BATT_AMP_PERVLT`. Then `MOT_BAT_CURR_MAX` can be set from the ESC's real rating. | Within 5 % |

Only after step 12 does the tracker command the drone: first watch-only (hover, tracker in `--no-fly`
with the camera, check detections and lock), then GUIDED with a walking person.

## How control passes between pilot and tracker

1. Arm (CH5), take off in **Loiter** (CH6 middle), climb to 6 m.
2. Flip CH6 to **Guided**: the tracker now holds position (no target) and commands nothing else.
3. Lock with **CH7**; the drone follows. **CH8** releases (drone holds).
4. **Any** other mode on CH6 (Loiter/AltHold) takes control back immediately. **CH10** = RTL.
5. Radio lost → RTL. Battery low → RTL; critical → land. Fence (200 m / 40 m) → RTL.
   The tracker's own leash stops it flying more than 150 m from where the flight began.

## Things this file deliberately does not do
- It does not skip any arming check: fix the cause of a PreArm message instead.
- It does not set PID gains: AutoTune does, on this airframe.
- `MOT_BAT_CURR_MAX 120` is a conservative guess until the ESC rating is known (the MVP hovers at about 16 A total).
