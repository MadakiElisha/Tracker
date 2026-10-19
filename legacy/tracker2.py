#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Phase 4: Integrated Tracker

 Wires together:
   perception.py  → YOLO11n + ByteTrack
   pid_controller → yaw + forward PID loops
   fc_interface   → ArduPilot / PX4 / iNav / Betaflight

 Usage:
   python3 tracker.py --fc ardupilot --connect 127.0.0.1:14550
   python3 tracker.py --fc ardupilot --connect 127.0.0.1:14550 --source track_test_clip.mp4
   python3 tracker.py --fc ardupilot --connect 127.0.0.1:14550 --no-fly

 Keyboard controls (video window must be focused):
   L  — lock on largest visible target
   R  — release lock
   T  — arm and takeoff (if not airborne)
   H  — return to home (RTL)
   Q  — quit and RTL
   +  — increase follow distance (smaller setpoint area)
   -  — decrease follow distance (larger setpoint area)
=============================================================
"""

import cv2
import numpy as np
import time
import argparse
import threading
import sys
from collections import defaultdict
from pathlib import Path

from ultralytics import YOLO
from pid_controller2 import TrackerPID
from fc_interface3dv import FCInterface

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL       = "yolo11n.pt"
FRAME_W          = 640
FRAME_H          = 480
CONF_THRESHOLD   = 0.45
IOU_THRESHOLD    = 0.45
MAX_LOST_FRAMES  = 25
TARGET_ALTITUDE  = 10.0      # metres AGL for takeoff

TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

CLASS_COLORS = {
    0: (0,   255,  0  ),
    2: (255, 100,  0  ),
    3: (0,   200,  255),
    5: (128, 0,    255),
    7: (0,   80,   255),
}

# ============================================================
# GLOBAL STATE
# ============================================================
locked_id        = None
lost_frames      = defaultdict(int)
last_boxes       = {}

# Commands from PID (written by main thread, read by FC thread)
cmd_vx           = 0.0
cmd_yaw          = 0.0
cmd_lock         = threading.Lock()

# Telemetry from FC (written by FC thread, read by draw thread)
telemetry        = {}
telem_lock       = threading.Lock()

# Control flags
flag_takeoff     = threading.Event()
flag_land        = threading.Event()
flag_quit        = threading.Event()
is_airborne      = False

# ============================================================
# FC COMMAND THREAD
# Runs at 20Hz, decoupled from vision loop
# ============================================================
def fc_thread_fn(fc, no_fly: bool):
    global is_airborne, cmd_vx, cmd_yaw

    print("[FC THREAD] Started.")

    while not flag_quit.is_set():

        # --- Takeoff request ---
        if flag_takeoff.is_set() and not is_airborne:
            flag_takeoff.clear()
            if not no_fly:
                try:
                    fc.arm_and_takeoff(TARGET_ALTITUDE)
                    is_airborne = True
                    print("[FC THREAD] Airborne.")
                except Exception as e:
                    print(f"[FC THREAD] Takeoff failed: {e}")
            else:
                print("[FC THREAD] --no-fly active, skipping takeoff.")
                is_airborne = True   # pretend airborne for testing

        # --- Land request ---
        if flag_land.is_set():
            flag_land.clear()
            if not no_fly:
                fc.land()
            is_airborne = False
            with cmd_lock:
                cmd_vx = 0.0
                cmd_yaw = 0.0
            print("[FC THREAD] Landing.")

        # --- Send velocity commands if airborne ---
        if is_airborne:
            with cmd_lock:
                vx  = cmd_vx
                yaw = cmd_yaw
            if not no_fly:
                try:
                    fc.send_velocity(vx, yaw)
                except Exception as e:
                    print(f"[FC THREAD] Send velocity error: {e}")

        # --- Read telemetry ---
        try:
            t = fc.get_telemetry()
            # FIX: Only attempt updates if telemetry is a valid dictionary to avoid TypeError
            if t is not None and isinstance(t, dict):
                with telem_lock:
                    telemetry.update(t)
        except Exception as e:
            # Shield thread from failing if connection drops temporarily
            pass

        time.sleep(0.05)   # 20 Hz

    print("[FC THREAD] Stopped.")


# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_tracks(frame, tracks, locked_id):
    fh, fw = frame.shape[:2]
    cx_f   = fw // 2
    cy_f   = fh // 2

    cv2.drawMarker(frame, (cx_f, cy_f),
                   (255, 255, 255), cv2.MARKER_CROSS, 24, 1)

    locked_box = None

    for t in tracks:
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        tid   = int(t[4])
        conf  = float(t[5])
        cid   = int(t[6])

        is_locked  = (tid == locked_id)
        color      = CLASS_COLORS.get(cid, (200, 200, 200))
        cls_name   = TARGET_CLASSES.get(cid, "object")

        if is_locked:
            draw_col  = (0, 255, 255)
            thickness = 3
            locked_box = (x1, y1, x2, y2)
        else:
            draw_col  = color
            thickness = 2

        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_col, thickness)

        # Corner accents on locked target
        if is_locked:
            L = 18
            for (px, py, dx, dy) in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (px, py), (px + dx*L, py), draw_col, 2)
                cv2.line(frame, (px, py), (px, py + dy*L), draw_col, 2)

        # Label
        label    = f"{'[LOCK] ' if is_locked else ''}ID:{tid} {cls_name} {conf:.2f}"
        lbl_w, lbl_h = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.rectangle(frame,
                      (x1, y1 - lbl_h - 6),
                      (x1 + lbl_w + 4, y1),
                      draw_col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 0), 1, cv2.LINE_AA)

        # Centre dot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, draw_col, -1)

        # Error arrow to locked target
        if is_locked:
            cv2.arrowedLine(frame,
                            (cx_f, cy_f), (cx, cy),
                            (0, 255, 255), 1, tipLength=0.12)

    return frame, locked_box


def draw_hud(frame, locked_id, locked_box,
             track_count, fps, pid, no_fly):
    fh, fw = frame.shape[:2]

    with cmd_lock:
        vx  = cmd_vx
        yaw = cmd_yaw
    with telem_lock:
        t = dict(telemetry)

    # --- Left panel background ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (230, 185), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # FIX: Robust key fallback checking to match multiple flight controller formats
    alt  = t.get('alt', t.get('altitude', 0.0))
    mode = t.get('mode', t.get('flight_mode', '?'))
    hdg  = t.get('heading', t.get('hdg', 0))
    bat  = t.get('battery', t.get('voltage', t.get('bat', 0.0)))
    sats = t.get('gps_sats', t.get('sats', t.get('satellites', 0)))

    # FIX: Translate raw numeric modes from ArduPilot to clean strings if applicable
    ardupilot_mode_map = {
        0: "STABILIZE", 1: "ACRO", 2: "ALT_HOLD", 3: "AUTO", 4: "GUIDED",
        5: "LOITER", 6: "RTL", 7: "CIRCLE", 9: "LAND", 11: "DRIFT",
        16: "POSHOLD", 17: "BRAKE", 19: "SIMPLE"
    }
    if isinstance(mode, (int, float)) or (isinstance(mode, str) and mode.isdigit()):
        mode = ardupilot_mode_map.get(int(mode), f"MODE:{mode}")

    hud_lines = [
        f"FPS    : {fps:.1f}",
        f"Tracks : {track_count}",
        f"Lock   : {locked_id if locked_id is not None else 'NONE'}",
        f"ALT    : {alt:.1f}m",
        f"HDG    : {hdg}deg",
        f"MODE   : {mode}",
        f"BAT    : {bat:.1f}V",
        f"SATS   : {sats}",
    ]

    if locked_box is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh)
        hud_lines += [
            f"Vx     : {vx:+.2f} m/s",
            f"Yaw    : {yaw:+.2f} r/s",
            f"ErrX   : {dbg['error_x']:+.3f}",
            f"Area   : {dbg['current_area']:.3f}/{dbg['setpoint_area']:.3f}",
        ]

    for i, line in enumerate(hud_lines):
        cv2.putText(frame, line, (10, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # --- Status badge ---
    if locked_id is not None:
        status      = "TRACKING"
        status_col  = (0, 255, 0)
    else:
        status      = "SCANNING"
        status_col  = (0, 165, 255)

    if not is_airborne:
        status     = "GROUNDED"
        status_col = (0, 0, 255)

    cv2.putText(frame, status, (fw - 140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                status_col, 2, cv2.LINE_AA)

    if no_fly:
        cv2.putText(frame, "[NO-FLY MODE]", (fw - 165, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 80, 255), 1, cv2.LINE_AA)

    # --- Distance bar (right edge) ---
    if locked_box is not None:
        dbg      = pid.get_debug_info(locked_box, fw, fh)
        bar_x    = fw - 28
        bar_top  = 70
        bar_bot  = fh - 70
        bar_h    = bar_bot - bar_top
        fill_frac = min(dbg['current_area'] / max(dbg['setpoint_area'] * 2, 0.001), 1.0)
        fill_h   = int(fill_frac * bar_h)

        cv2.rectangle(frame,
                      (bar_x, bar_top),
                      (bar_x + 14, bar_bot),
                      (50, 50, 50), -1)
        bar_col = ((0, 255, 0) if abs(dbg['error_area']) < 0.02 else (0, 140, 255))
        cv2.rectangle(frame,
                      (bar_x, bar_bot - fill_h),
                      (bar_x + 14, bar_bot),
                      bar_col, -1)
        cv2.putText(frame, "DIST",
                    (bar_x - 4, bar_top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 200), 1)

    # --- Controls footer ---
    footer = "[L]Lock [R]Release [T]Takeoff [H]Home [+/-]Dist [Q]Quit"
    cv2.putText(frame, footer, (5, fh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# ============================================================
# MAIN
# ============================================================
def main():
    global locked_id, cmd_vx, cmd_yaw, is_airborne

    parser = argparse.ArgumentParser(
        description='Drone Tracker — Integrated')
    parser.add_argument('--fc',
        default='ardupilot',
        help='FC type: ardupilot|px4|inav|betaflight')
    parser.add_argument('--connect',
        default='127.0.0.1:14550',
        help='FC connection string')
    parser.add_argument('--source',
        default='0',
        help='Camera index (0) or video file path')
    parser.add_argument('--no-fly',
        action='store_true',
        help='Run perception and PID without sending FC commands')
    parser.add_argument('--altitude',
        type=float,
        default=10.0,
        help='Takeoff altitude in metres')
    args = parser.parse_args()

    global TARGET_ALTITUDE
    TARGET_ALTITUDE = args.altitude

    # --- Connect to FC ---
    print(f"[TRACKER] Connecting to FC ({args.fc})...")
    fc = FCInterface(args.fc, args.connect)
    try:
        fc.connect()
    except Exception as e:
        print(f"[TRACKER] FC connection failed: {e}")
        if not args.no_fly:
            sys.exit(1)
        print("[TRACKER] Continuing in --no-fly mode.")

    # --- Start FC command thread ---
    fc_thread = threading.Thread(
        target=fc_thread_fn,
        args=(fc, args.no_fly),
        daemon=True)
    fc_thread.start()

    # --- Load YOLO ---
    print("[TRACKER] Loading YOLO11n...")
    model = YOLO(YOLO_MODEL)
    pid   = TrackerPID()

    # --- Open video source ---
    src = args.source
    if src.isdigit():
        src = int(src)
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[TRACKER] Cannot open source: {args.source}")
        print("[TRACKER] Falling back to test pattern.")
        cap = None

    print("[TRACKER] Running.")
    print("  L = lock  R = release  T = takeoff")
    print("  H = home  +/- = dist   Q = quit")

    fps    = 0.0
    t_prev = time.time()

    try:
        while not flag_quit.is_set():

            # --- Frame acquisition ---
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    # Loop video file
                    if isinstance(src, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            else:
                frame = np.full(
                    (FRAME_H, FRAME_W, 3), 60, dtype=np.uint8)
                tx = int(FRAME_W/2 + 150*np.sin(time.time()*0.4))
                cv2.rectangle(frame,
                              (tx-35, 120), (tx+35, 340),
                              (0, 0, 200), -1)

            # --- FPS ---
            t_now = time.time()
            dt    = max(t_now - t_prev, 1e-4)
            fps   = 0.9*fps + 0.1*(1.0/dt)
            t_prev = t_now

            # --- YOLO tracking ---
            results = model.track(
                frame,
                persist  = True,
                verbose  = False,
                conf     = CONF_THRESHOLD,
                iou      = IOU_THRESHOLD,
                classes  = list(TARGET_CLASSES.keys()),
                tracker  = "bytetrack.yaml",
            )

            # --- Parse detections ---
            tracks = []
            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for i in range(len(ids)):
                    tid = ids[i]
                    entry = [
                        boxes[i][0], boxes[i][1],
                        boxes[i][2], boxes[i][3],
                        tid, confs[i], classes[i]
                    ]
                    tracks.append(entry)
                    last_boxes[tid] = entry

                active_ids = set(ids)
                for tid in list(lost_frames.keys()):
                    if tid not in active_ids:
                        lost_frames[tid] += 1
                        if lost_frames[tid] > MAX_LOST_FRAMES:
                            if tid == locked_id:
                                print(f"\n[TRACKER] Lock lost: ID {tid} gone for {MAX_LOST_FRAMES} frames.")
                                locked_id = None
                                pid.reset()
                                with cmd_lock:
                                    cmd_vx  = 0.0
                                    cmd_yaw = 0.0
                            lost_frames.pop(tid, None)
                            last_boxes.pop(tid, None)
                    else:
                        lost_frames[tid] = 0

            # --- PID update ---
            locked_box = None
            if locked_id is not None:
                # Find locked track in current detections
                locked_track = next(
                    (t for t in tracks if int(t[4]) == locked_id),
                    None)

                if locked_track is not None:
                    locked_box = (
                        int(locked_track[0]), int(locked_track[1]),
                        int(locked_track[2]), int(locked_track[3]))
                    vx, yaw = pid.compute(
                        locked_box, FRAME_W, FRAME_H)
                    with cmd_lock:
                        cmd_vx  = vx
                        cmd_yaw = yaw
                else:
                    # Target temporarily invisible — coast then stop
                    with cmd_lock:
                        cmd_vx  *= 0.8
                        cmd_yaw *= 0.8
            else:
                # No lock — hover
                with cmd_lock:
                    cmd_vx  = 0.0
                    cmd_yaw = 0.0
                pid.reset()

            # --- Draw ---
            frame, locked_box = draw_tracks(frame, tracks, locked_id)
            frame = draw_hud(
                frame, locked_id, locked_box,
                len(tracks), fps, pid, args.no_fly)

            cv2.imshow("Drone Tracker", frame)

            # --- Keyboard ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                flag_quit.set()

            elif key == ord('l'):
                if tracks:
                    best = max(tracks,
                               key=lambda t: (t[2]-t[0])*(t[3]-t[1]))
                    locked_id = int(best[4])
                    pid.reset()
                    print(f"\n[LOCK] ID:{locked_id} ({TARGET_CLASSES.get(int(best[6]),'?')})")
                else:
                    print("\n[LOCK] No targets visible.")

            elif key == ord('r'):
                print(f"\n[LOCK] Released (was ID:{locked_id})")
                locked_id = None
                pid.reset()
                with cmd_lock:
                    cmd_vx  = 0.0
                    cmd_yaw = 0.0

            elif key == ord('t'):
                print("\n[CMD] Takeoff requested.")
                flag_takeoff.set()

            elif key == ord('h'):
                print("\n[CMD] RTL requested.")
                flag_land.set()

            elif key == ord('+') or key == ord('='):
                pid.setpoint_area = max(
                    0.02, pid.setpoint_area - 0.01)
                print(f"\n[PID] Follow distance increased (setpoint={pid.setpoint_area:.2f})")

            elif key == ord('-'):
                pid.setpoint_area = min(
                    0.50, pid.setpoint_area + 0.01)
                print(f"\n[PID] Follow distance decreased (setpoint={pid.setpoint_area:.2f})")

    finally:
        print("\n[TRACKER] Shutting down...")
        flag_quit.set()
        with cmd_lock:
            cmd_vx  = 0.0
            cmd_yaw = 0.0
        if not args.no_fly and is_airborne:
            print("[TRACKER] Triggering RTL...")
            fc.land()
        fc_thread.join(timeout=3)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[TRACKER] Stopped.")


if __name__ == '__main__':
    main()
