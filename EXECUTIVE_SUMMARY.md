> **Status update (latest build):** config-driven PID, safety supervisor, hardened FC layer, run logging/analyzer, offline closed-loop model, Gazebo camera patch and `gz:` source are implemented; 172 tests pass. NOT yet done: live Gazebo end-to-end run, VisDrone `best.pt` download/test, real hardware. See docs/SIM_VALIDATION.md. Earlier status statements below may be stale.

# DRONE TRACKER SYSTEM
## Executive Summary, Hardware Specification & Use Case Reference

**Document Version:** 1.0
**Project Status:** Active Development — Phase 5 Complete, Phase 6 Pending Hardware
**Classification:** Internal Reference Document

---

## 1. EXECUTIVE SUMMARY

### What This Is

The Drone Tracker is a fully autonomous, AI-powered target detection, locking, and
follow system designed for deployment on FPV (First Person View) drones. The system
enables a drone to independently identify, lock onto, and pursue a designated
target — a person, vehicle, or other object — without continuous manual pilot
intervention, while the pilot retains full override authority at all times.

The system is inspired by and designed to match the core operational capabilities of
systems such as the Skydio X10 and Anduril Bolt — specifically their autonomous
detection, target locking, and persistent following capabilities — excluding any
offensive or weaponised functionality. This is a pure surveillance, tracking, and
follow platform.

### What Has Been Built

A complete software stack running on a companion computer aboard the drone,
comprising six integrated components:

1. **Perception Pipeline** — Multi-class object detector (YOLO11n) combined with a
   multi-object tracker (ByteTrack) that assigns persistent identities to every
   detected object across frames, with Kalman filter prediction during occlusion.

2. **Flight Controller Interface** — A hardware-agnostic abstraction layer that
   communicates with ArduPilot, PX4, iNav, and Betaflight flight controllers via
   MAVLink or MSP protocol, with a watchdog failsafe and queue-based command
   architecture.

3. **3-Axis PID Controller** — A three-axis proportional-integral-derivative
   controller that converts target position in the camera frame into velocity
   commands: yaw (left/right centering), pitch (forward/backward distance
   maintenance), and Z (altitude adjustment to keep target vertically centred).

4. **Target Assignment System** — Three independent target designation methods
   operating simultaneously: RC transmitter switch (hardware), browser-based
   GCS tap-to-lock (software over WiFi), and mouse click on the local display.

5. **Ground Control Station Server** — A lightweight browser-accessible video
   streaming server that broadcasts the annotated tracking feed to any device
   on the network and accepts tap coordinates for target designation.

6. **Simulation Environment** — A full Gazebo Harmonic + ArduPilot SITL simulation
   with moving targets and an occlusion wall for development and validation.

### Current Status

```
Phase 1: Simulation Environment         COMPLETE ✅
Phase 2: Perception Pipeline            COMPLETE ✅
Phase 3: FC Abstraction Layer           COMPLETE ✅
Phase 4: 3-Axis PID Controller         COMPLETE ✅
Phase 5: Target Assignment              COMPLETE ✅
Phase 6: Hardware Deployment Guides     COMPLETE ✅ (awaiting hardware)
Phase 7: iNav/Betaflight MSP            PENDING ⏳
Phase 8: Appearance-Based Re-ID         PENDING ⏳
Phase 9: OSD Goggles Integration        PENDING ⏳
```

### What Is Not Yet Done

- End-to-end sim validation of the full flying follow loop
- Physical hardware procurement and first deployment
- iNav and Betaflight MSP RC injection backends
- Appearance-based Re-ID for automatic re-lock after full occlusion

---

## 2. IMPORTANCE AND USE CASES

### Why This System Matters

Traditional drone surveillance and tracking requires a skilled pilot providing
continuous manual control — an operator who must simultaneously fly the drone,
maintain awareness of the target, and avoid obstacles. This approach has four
fundamental limitations: it requires a trained pilot at all times, the pilot's
attention is divided between flying and tracking, fatigue reduces performance over
time, and the drone cannot act on anything unless the pilot is actively engaged.

This system eliminates the pilot's tracking burden entirely. The drone autonomously
handles target lock and pursuit. The pilot's role shifts from active tracking to
mission oversight — they can intervene, redirect, or override at any moment, but
they are not required to continuously track the target manually.

### Primary Use Cases

**1. Security and Perimeter Surveillance**
A security operator designates a person of interest via the GCS tap interface.
The drone autonomously follows that individual across a perimeter, maintaining
a constant follow distance, adjusting altitude to keep the target in view, and
re-acquiring automatically after partial occlusion by structures. The operator
monitors the GCS feed and intervenes only when necessary.

**2. Search and Rescue Operations**
In an SAR scenario, a drone equipped with this system can be dispatched to a
last-known location. When the perception system identifies a person, the operator
locks onto them and the drone maintains position relative to that person while
the ground team navigates toward the location using the GCS feed as a reference.

**3. Intelligence, Surveillance, and Reconnaissance (ISR)**
The system can track and follow vehicles or individuals across terrain without
requiring a dedicated drone pilot for the duration of the mission. A single
operator can monitor multiple drones, each autonomously managing its own target.

**4. Convoy and Asset Escort**
A vehicle convoy can be escorted by a drone that autonomously maintains position
above a designated lead vehicle, providing aerial coverage without a dedicated
pilot for each drone in the escort.

**5. Wildlife and Environmental Monitoring**
Research applications where tracking an animal or vehicle across open terrain
for extended periods would be impossible with manual piloting but straightforward
with autonomous following.

**6. Film, Sport, and Event Coverage**
The RC switch lock-on enables a single camera operator to designate a subject
and have the drone autonomously maintain a cinematic follow shot, freeing the
operator to manage framing and timing rather than flight control.

**7. Tactical Ground Support**
Ground personnel can designate a target of interest on their tablet via the GCS
interface. The drone follows and provides continuous aerial observation without
requiring a dedicated drone operator in the team.

---

## 3. HARDWARE SPECIFICATION

### 3.1 Companion Computer — Two Options

#### Option A: Raspberry Pi 5 8GB + Hailo-8L AI Hat (RECOMMENDED FOR FIRST DEPLOYMENT)

| Component | Specification | Approx. Cost |
|-----------|---------------|--------------|
| Raspberry Pi 5 8GB | ARM Cortex-A76 × 4, 8GB LPDDR4X | $80 USD |
| Raspberry Pi AI Hat+ | Hailo-8L NPU, 26 TOPS via PCIe M.2 | $70 USD |
| MicroSD 64GB (A2 rated) | Samsung Pro Endurance recommended | $15 USD |
| Official RPi 5 PSU (5V/5A) | For bench setup only | $12 USD |
| **Total** | | **~$177 USD** |

**Performance:** YOLO11n at 30-45 FPS, 22-33ms latency
**Weight:** ~80g (board + hat + camera)
**Power draw:** ~10-11W at full load

Why first: Lower cost, lighter, easier to integrate mechanically on an FPV drone,
and the Hailo-8L delivers flight-capable inference speed for our target model.

#### Option B: Nvidia Jetson Orin Nano 8GB Developer Kit

| Component | Specification | Approx. Cost |
|-----------|---------------|--------------|
| Jetson Orin Nano 8GB Dev Kit | 6-core ARM + 1024 CUDA cores | $499 USD |
| MicroSD 64GB or NVMe SSD | Samsung 970 Evo recommended | $20-80 USD |
| **Total** | | **~$519-579 USD** |

**Performance:** YOLO11n at 80-120 FPS (TensorRT FP16), 8-12ms latency
**Weight:** ~200g with dev kit carrier board; ~60g with compact carrier
**Power draw:** ~10-15W at full load

Why second: Higher performance ceiling, better for multi-camera or larger
model variants, lower latency for high-speed tracking scenarios.

---

### 3.2 Tracking Cameras — Complete Options

#### The Critical Requirement: Global Shutter

This is the single most important camera specification for this application.

A **rolling shutter** camera reads its sensor line by line from top to bottom.
When the drone or target is moving rapidly, the lines are captured at slightly
different moments in time, causing the image to appear skewed, wobbled, or bent.
This is the "jello effect" familiar to FPV pilots. For human vision, the brain
compensates. For YOLO, it does not — a distorted frame causes bounding boxes to
jump, detection confidence to drop, and ByteTrack to lose track ID consistency.

A **global shutter** camera reads its entire sensor simultaneously at one instant.
Every pixel in the frame represents the exact same moment in time regardless of
how fast the drone is moving. Detections are clean, bounding boxes are stable,
and tracking is consistent through aggressive manoeuvres.

**For any serious deployment: global shutter is mandatory.**

---

#### OPTION 1 — Raspberry Pi AI Camera (IMX500) ⭐ RECOMMENDED FOR RPi 5

| Specification | Value |
|---------------|-------|
| Sensor | Sony IMX500 |
| Shutter Type | **Global Shutter** ✅ |
| Resolution | 12MP (4056×3040) |
| Max FPS at tracking res | 60 FPS at 1080p; 120 FPS at 720p |
| Interface | CSI-2 (MIPI) |
| Connection | Directly to RPi 5 via AI Hat+ CSI port |
| Onboard AI | 2 TOPS (unused by our system — Hailo handles inference) |
| FOV | ~66° diagonal (standard lens) |
| Weight | ~6g |
| Approx. Cost | $80 USD |

**Why this:** Purpose-built for RPi 5 + AI Hat stack. CSI interface means
zero-latency direct feed to the companion computer. Global shutter. Lightweight.
No additional adapters or capture cards required.

**Integration:** Connects directly to the CSI port on the Hailo AI Hat+ (not
the RPi 5 directly when the hat is stacked). Uses Picamera2 Python library.

**Note on the IMX500's onboard AI:** The sensor has a built-in 2 TOPS AI
processor capable of running small networks directly on the chip. We do not
use this — our full YOLO11n pipeline runs on the Hailo-8L at 26 TOPS, which
is far more capable. The IMX500 AI chip is simply unused.

---

#### OPTION 2 — Arducam IMX477 Global Shutter ⭐ RECOMMENDED FOR JETSON

| Specification | Value |
|---------------|-------|
| Sensor | Sony IMX477 |
| Shutter Type | **Global Shutter** ✅ |
| Resolution | 12.3MP (4056×3040) |
| Max FPS at tracking res | 60 FPS at 1080p |
| Interface | CSI-2 (MIPI) |
| Connection | CSI port on Jetson Orin Nano/NX carrier board |
| FOV | Lens dependent (15-70° options available) |
| Weight | ~15-25g depending on lens |
| Approx. Cost | $65-85 USD (without lens) |

**Why this:** The gold standard for Jetson-based machine vision on drones.
Used by research groups and commercial developers on Jetson platforms.
The IMX477 has excellent sensitivity and low rolling shutter (global).
Available with multiple lens options for different FOV requirements.

**Integration:** Connects to the CAM0 or CAM1 CSI port on the Jetson carrier
board. Accessed via GStreamer pipeline using the NVIDIA NvArgus camera stack,
which provides hardware-accelerated capture direct to GPU memory.

---

#### OPTION 3 — Arducam IMX296 Global Shutter (Lightweight Alternative)

| Specification | Value |
|---------------|-------|
| Sensor | Sony IMX296 |
| Shutter Type | **Global Shutter** ✅ |
| Resolution | 1.58MP (1456×1088) |
| Max FPS at tracking res | 120 FPS at full resolution |
| Interface | CSI-2 |
| Weight | ~8g |
| Approx. Cost | $35-45 USD |

**When to use:** Weight is a hard constraint (sub-250g build), or you need
higher FPS for very fast-moving targets. The lower resolution means smaller
detection range, but at higher frame rates the tracker compensates.

---

#### WHAT NOT TO USE: Rolling Shutter Cameras

These are explicitly not recommended for this application:

| Camera | Problem |
|--------|---------|
| Runcam, Foxeer, Caddx analog FPV cams | Rolling shutter + analog signal requires capture card |
| Standard RPi Camera Module 2 | Rolling shutter sensor (IMX219) |
| Logitech webcams | Rolling shutter, USB latency, not drone-rated |
| GoPro via HDMI capture | Rolling shutter, capture latency |
| DJI O3 via USB | Rolling shutter, USB video encoding adds 200-300ms |

---

### 3.3 FPV Pilot Camera (Separate — Pilot Video Only)

This camera is entirely independent of the tracking system. It goes to the
VTX and into the pilot's goggles. The tracker does not touch this feed.

| Camera | Type | Why |
|--------|------|-----|
| Foxeer Razer Nano 1200TVL | Analog | Lightest, wide DR, smooth |
| Runcam Phoenix 2 Nano | Analog | Excellent low-light |
| Caddx Ratel 2 | Analog | Strong low-light, compact |
| DJI O3 Air Unit | Digital | Best image quality, built-in DVR |
| Walksnail Avatar HD Nano | Digital | Good digital alternative to O3 |

For follow-me missions at moderate speed, any of these works fine.
For high-speed FPV the pilot camera is critical for situational awareness.

---

### 3.4 Flight Controller

The FC runs the flight stabilisation and responds to our velocity commands
via MAVLink. Selection depends on target firmware.

#### ArduPilot / PX4 (RECOMMENDED — Phase 5 FC interface is built for these)

| FC | Firmware | Why |
|----|----------|-----|
| **Pixhawk 6C Mini** | ArduPilot / PX4 | Industry standard, small form factor, excellent MAVLink support |
| **Matek H743-Mini** | ArduPilot | Lightweight, powerful F7 processor, budget-friendly |
| **CUAV X7+ Pro** | ArduPilot / PX4 | Redundant sensors, high-end missions |
| **Holybro Kakute H7 v2** | ArduPilot / PX4 | Budget option, FPV-native form factor |
| **SpeedyBee F405 Wing** | ArduPilot / iNav | Good iNav support for Phase 7 |

#### iNav / Betaflight (Phase 7 — MSP injection, not yet implemented)

| FC | Firmware | Notes |
|----|----------|-------|
| Kakute H7 v2 | iNav | Dual-use — also runs ArduPilot |
| Mamba F722 | Betaflight / iNav | Budget, good community support |
| SpeedyBee F405 V4 | Betaflight / iNav | Popular FPV FC |

**UART requirement:** Whichever FC is chosen must have one free UART
(serial port) for the MAVLink or MSP connection to the companion computer.
Most FCs have 3-6 UARTs — this is not a constraint in practice.

---

### 3.5 Drone Frame

The frame must accommodate the companion computer, camera, and cabling
without compromising flight performance.

| Frame | Size | AUW Target | Use Case |
|-------|------|------------|----------|
| iFlight Chimera7 Pro V2 | 7" | 600-800g | Best balance — enough lift for companion computer, good flight time |
| TBS Source One v5 | 5" | 400-550g | Lighter, faster, tighter payload constraint |
| iFlight Nazgul Evoque F5 | 5" | 400-500g | Performance FPV with payload space |
| iFlight Chimera10 | 10" | 900-1200g | Maximum payload capacity, longer flight time |

**For RPi 5 + Hailo stack (~115g payload):**
A 7" frame is the minimum comfortable choice. 5" is achievable but tight.

**For Jetson Orin Nano compact module (~60g payload):**
A 5" frame is viable. Developer kit form factor requires 7"+.

---

### 3.6 RC Transmitter and Receiver

The RC system must support at least 9 channels to use all three
assignment control channels (CH7, CH8, CH9).

| Component | Recommendation | Why |
|-----------|----------------|-----|
| Transmitter | RadioMaster Boxer / TX16S (EdgeTX) | 16-channel, excellent switch layout, open firmware |
| Protocol | ELRS (ExpressLRS) | Long range, low latency, open source |
| Receiver | RadioMaster RP4TD (ELRS) | Redundant antennas, small, reliable |
| Alternative TX | FrSky X20S / Tandem X20S | ACCST/ACCESS, good failsafe handling |

**Channel assignments for this system:**
- CH1-CH4: Standard roll/pitch/throttle/yaw
- CH7: 2-position momentary switch → lock on largest target
- CH8: 2-position toggle switch → release lock
- CH9: 3-position switch → class filter (Any / Person / Vehicle)

---

### 3.7 Video Transmission (FPV to Goggles)

Completely independent of the tracker. Standard FPV VTX chain.

| Component | Recommendation |
|-----------|----------------|
| VTX (Analog) | TBS Unify Pro32 Nano / Foxeer Reaper Nano |
| VTX (Digital) | DJI O3 Air Unit / Walksnail Avatar HD Nano |
| Goggles (Analog) | FatShark Dominator HD / Skyzone Cobra X |
| Goggles (Digital) | DJI Goggles 3 / Walksnail Goggles X |

**For tracking missions:** Digital FPV is preferred because the operator
needs to make precise identification of targets at distance. Digital systems
(DJI O3, Walksnail) provide substantially better image clarity at range.

---

### 3.8 Power System

| Component | Specification | Notes |
|-----------|---------------|-------|
| Battery | 6S LiPo, 3000-5000mAh (7") | Higher capacity = longer mission time |
| BEC for companion computer | 5V/3A minimum, 5V/5A recommended | Must supply RPi 5 peak current |
| BEC capacitor | 1000µF 16V electrolytic inline | Absorbs ESC switching noise |
| ESCs | 4-in-1 or individual, rated for motor Kv | Standard FPV ESC |

**CRITICAL:** The RPi 5 requires a stable 5V supply. It will brown-out and
corrupt its SD card if voltage dips below 4.7V during motor spin-up transients.
Use a dedicated regulated BEC, not a shared rail with the FC.

---

### 3.9 Communication Links Summary

| Link | Hardware | Software |
|------|----------|----------|
| Companion → FC (MAVLink) | UART GPIO or USB-UART adapter | pymavlink in fc_interface5.py |
| GCS browser | WiFi (built-in RPi 5) or separate WiFi adapter | Flask/waitress server on port 8080 |
| Telemetry to QGC | MAVProxy UDP forwarding | Already configured in launch_sim.sh |
| FPV video | VTX → goggles (RF) | Independent of tracker |
| RC control | Transmitter → ELRS receiver → FC | Standard FPV RC |

---

## 4. CAMERA INTEGRATION DETAILS

### How the Tracking Camera Connects

```
[Tracking Camera]
      │
      │ CSI ribbon cable
      ▼
[AI Hat+ / Jetson CSI port]
      │
      │ Internal PCIe / MIPI bus
      ▼
[Companion Computer CPU]
      │
      │ In-memory frame buffer
      ▼
[YOLO11n Inference]  → (Hailo-8L via PCIe) or (TensorRT on Jetson GPU)
      │
      ▼
[Annotated frames]
      ├── OpenCV display window (local, if monitor attached)
      └── GCS server → WiFi → browser on any device
```

### Latency Stack — End to End

| Stage | RPi 5 + Hailo | Jetson Orin Nano |
|-------|---------------|-----------------|
| Camera sensor exposure | ~8ms | ~8ms |
| CSI transfer to CPU | ~2ms | ~2ms |
| YOLO inference | ~25ms | ~10ms |
| ByteTrack update | ~2ms | ~1ms |
| PID compute | <1ms | <1ms |
| MAVLink send | ~2ms | ~2ms |
| FC processing + motor response | ~20ms | ~20ms |
| **Total** | **~60ms** | **~43ms** |

At 3 m/s follow speed, 60ms latency means the drone reacts to a position
that is approximately 18cm old. This is well within acceptable tolerance
for person or vehicle-speed tracking.

### Why Two Cameras (Not One)

The FPV pilot camera and the tracking camera serve fundamentally different
purposes with incompatible requirements:

| | Pilot Camera | Tracking Camera |
|-|--------------|-----------------|
| Purpose | Pilot situational awareness | Machine vision input |
| Latency requirement | <15ms (critical) | <100ms (acceptable) |
| FOV | Wide (120-150°) | Moderate (60-90°) |
| Shutter | Any (pilot compensates) | Global shutter required |
| Output format | Analog RF or digital compressed | Raw CSI frames |
| Failure consequence | Pilot loses orientation (critical) | Tracker loses target (recoverable) |

Combining them creates a coupled failure mode where a tracker software
issue could affect the pilot's video — an unacceptable safety constraint.

---

## 5. WHAT HAPPENS AT FIRST HARDWARE DEPLOYMENT

The first test sequence on real hardware:

1. **Bench test** (no props):
   - Connect RPi 5 to FC via USB-UART adapter
   - Run `python3 fc_interface5.py` — confirm heartbeat from FC
   - Run `python3 tracker.py --no-fly` with camera connected
   - Confirm bounding boxes on GCS browser
   - Confirm RC switches trigger lock/release events in terminal

2. **Props-off hover test** (FC on bench, motors spinning):
   - Run full tracker without --no-fly
   - Press T to arm — confirm FC arms
   - Check GCS browser for telemetry updates (alt, mode, armed)

3. **Tethered first flight** (drone physically constrained):
   - Tether the drone to prevent flyaway
   - Arm and takeoff to ~2m
   - Lock on a moving target (person walking past)
   - Observe PID commands in HUD — confirm drone attempts to follow

4. **Free flight** (open area, safety pilot on RC):
   - Follow a walking person in an open space
   - Test RC switch lock/release
   - Test GCS tap on tablet
   - Test occlusion (person walks behind vehicle)
   - Evaluate and tune PID gains for real hardware dynamics

---

## 6. OPEN ITEMS BEFORE REAL DEPLOYMENT

| Item | Priority | Effort |
|------|----------|--------|
| End-to-end sim validation (flying follow loop) | HIGH | 1 session |
| VisDrone model downloaded and tested in sim | HIGH | 1 session |
| PID gains tuned for real drone dynamics | HIGH | Field testing |
| iNav/Betaflight MSP backend (Phase 7) | MEDIUM | 2-3 sessions |
| Appearance Re-ID (Phase 8) | MEDIUM | 3-4 sessions |
| OSD text injection to goggles (Phase 9) | LOW | 1 session |
| Unit tests passing cleanly | MEDIUM | Done (tests written) |

---

*Document ends.*
*All hardware pricing approximate as of mid-2025. Verify current pricing before procurement.*
