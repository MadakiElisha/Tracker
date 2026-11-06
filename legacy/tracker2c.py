#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Integrated 3D Tracker (Phase 4 + Phase 5)

 Wires together:
   YOLO11n + ByteTrack     → perception
   pid_controller2.py      → 3-axis PID (yaw, pitch, Z)
   fc_interface3dv.py      → ArduPilot HAL (Now handles RC events)
   gcs_server.py           → Browser tap-to-lock
=============================================================
"""

import cv2
import numpy as np
import time
import argparse
import threading
import sys
from collections import defaultdict

from ultralytics import YOLO
from pid_controller2   import TrackerPID
from fc_interface4   import FCInterface
from gcs_server      import GCSServer [cite: 1, 2]

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL      = "yolo11n.pt"
FRAME_W         = 640
FRAME_H         = 480
CONF_THRESHOLD  = 0.45
IOU_THRESHOLD   = 0.45
MAX_LOST_FRAMES = 25
TARGET_ALTITUDE = 10.0 [cite: 2]

TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# RC CH9 class filter groups
CLASS_GROUPS = {
    'any':     [0, 2, 3, 5, 7],
    'person':  [0],
    'vehicle': [2, 3, 5, 7],
} [cite: 2]

CLASS_COLORS = {
    0: (0, 255, 0), 
    2: (255, 100, 0), 3: (0, 200, 255),
    5: (128, 0, 255), 7: (0, 80, 255),
} [cite: 2, 3]

# ============================================================
# GLOBAL STATE 
# ============================================================
locked_id   = None
lost_frames = defaultdict(int)
last_boxes = {} [cite: 3]

cmd_vx, cmd_vz, cmd_yaw = 0.0, 0.0, 0.0
cmd_lock = threading.Lock()

telemetry  = {}
telem_lock = threading.Lock()

flag_takeoff = threading.Event()
flag_land    = threading.Event()
flag_quit    = threading.Event()
is_airborne  = False [cite: 3]

# ============================================================
# FC COMMAND THREAD 
# ============================================================
def fc_thread_fn(fc, no_fly: bool):
    global is_airborne, cmd_vx, cmd_vz, cmd_yaw [cite: 3]

    print("[FC THREAD] Started.") [cite: 3]
    while not flag_quit.is_set():
        if flag_takeoff.is_set() and not is_airborne: [cite: 3, 4]
            flag_takeoff.clear()
            if not no_fly:
                try:
                    fc.arm_and_takeoff(TARGET_ALTITUDE)
                    is_airborne = True
                    print("[FC THREAD] Airborne.") [cite: 4, 5]
                except Exception as e: print(f"[FC THREAD] Takeoff failed: {e}")
            else: is_airborne = True

        if flag_land.is_set():
            flag_land.clear()
            if not no_fly: fc.land()
            is_airborne = False
            with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0 [cite: 5, 6]
            print("[FC THREAD] Landing.")

        if is_airborne and not no_fly:
            with cmd_lock: vx, vz, yaw = cmd_vx, cmd_vz, cmd_yaw [cite: 6]
            try: fc.send_velocity(vx, 0.0, vz, yaw)
            except Exception: pass

        try:
            t = fc.get_telemetry() [cite: 6, 7]
            with telem_lock: telemetry.update(t)
        except Exception: pass

        time.sleep(0.05)
    print("[FC THREAD] Stopped.") [cite: 7]


# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_tracks(frame, tracks, locked_id):
    fh, fw  = frame.shape[:2]
    cx_f, cy_f = fw // 2, fh // 2
    locked_box = None [cite: 7]

    cv2.drawMarker(frame, (cx_f, cy_f), (255, 255, 255), cv2.MARKER_CROSS, 24, 1) [cite: 7]

    for t in tracks: [cite: 7, 8]
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        tid, conf, cid = int(t[4]), float(t[5]), int(t[6])
        
        cls_name = TARGET_CLASSES.get(cid, "object")
        color    = CLASS_COLORS.get(cid, (200, 200, 200))
        is_lock  = (tid == locked_id) [cite: 8]

        if is_lock:
            draw_col, thickness = (0, 255, 255), 3 [cite: 8, 9]
            locked_box = (x1, y1, x2, y2)
        else:
            draw_col, thickness = color, 2 [cite: 9]

        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_col, thickness) [cite: 9]

        if is_lock:
            L = 18
            for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]: [cite: 9, 10]
                cv2.line(frame, (px, py), (px + dx*L, py), draw_col, 2)
                cv2.line(frame, (px, py), (px, py + dy*L), draw_col, 2)
            cv2.arrowedLine(frame, (cx_f, cy_f), ((x1 + x2) // 2, (y1 + y2) // 2), (0, 255, 255), 1, tipLength=0.12) [cite: 10]

        label = f"{'[LOCK] ' if is_lock else ''}ID:{tid} {cls_name} {conf:.2f}" [cite: 10, 11]
        lbl_w, lbl_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lbl_h - 6), (x1 + lbl_w + 4, y1), draw_col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 4, draw_col, -1) [cite: 11]

    return frame, locked_box [cite: 11]

def draw_hud(frame, locked_id, locked_box, track_count, fps, pid, no_fly, rc_filter): [cite: 11, 12]
    fh, fw = frame.shape[:2]
    with cmd_lock: vx, vz, yaw = cmd_vx, cmd_vz, cmd_yaw
    with telem_lock: t = dict(telemetry) [cite: 12]

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (230, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame) [cite: 12]

    mode = t.get('mode', '?')
    ardupilot_mode_map = {"0": "STABILIZE", "4": "GUIDED", "5": "LOITER", "6": "RTL", "9": "LAND"}
    mode_text = ardupilot_mode_map.get(str(mode), f"MODE:{mode}") [cite: 12]

    lines = [ [cite: 12]
        f"FPS    : {fps:.1f}", f"Tracks : {track_count}", f"Lock   : {locked_id if locked_id is not None else 'NONE'}", [cite: 13]
        f"ALT    : {t.get('alt', 0.0):.1f}m", f"HDG    : {t.get('heading', 0)}deg", f"MODE   : {mode_text}",
        f"BAT    : {t.get('battery', 0.0):.1f}V", f"SATS   : {t.get('gps_sats', 0)}",
        f"FLTR   : {rc_filter.upper()}"
    ] [cite: 13]

    if locked_box is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh) [cite: 13, 14]
        lines += [
            f"Vx     : {vx:+.2f} m/s", f"Vz     : {vz:+.2f} m/s", f"Yaw    : {yaw:+.2f} r/s",
            f"ErrX   : {dbg['error_x']:+.3f}", f"ErrY   : {dbg.get('error_y', 0):+.3f}",
            f"Area   : {dbg['current_area']:.3f}/{dbg['setpoint_area']:.3f}"
        ] [cite: 14]

    for i, line in enumerate(lines): [cite: 14, 15]
        cv2.putText(frame, line, (10, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA) [cite: 15]

    status, scol = ("GROUNDED", (0, 0, 255)) if not is_airborne else ("TRACKING", (0, 255, 0)) if locked_id is not None else ("SCANNING", (0, 165, 255)) [cite: 15]
    cv2.putText(frame, status, (fw - 140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, scol, 2, cv2.LINE_AA)
    if no_fly: cv2.putText(frame, "[NO-FLY MODE]", (fw - 165, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1, cv2.LINE_AA) [cite: 15]

    if locked_box is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh) [cite: 15, 16]
        bx, bt, bb = fw - 28, 70, fh - 70
        fill = int(min(dbg['current_area'] / max(dbg['setpoint_area'] * 2, 0.001), 1.0) * (bb - bt))
        bcol = (0, 255, 0) if abs(dbg['error_area']) < 0.02 else (0, 140, 255)
        cv2.rectangle(frame, (bx, bt), (bx+14, bb), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, bb-fill), (bx+14, bb), bcol, -1)
        cv2.putText(frame, "DIST", (bx-4, bt-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1) [cite: 16, 17]

    cv2.putText(frame, "[L]Lock [R]Rel [T]Takeoff [H]Home [+/-]Dist [Q]Quit", (5, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA) [cite: 17]
    return frame [cite: 17]

# ============================================================
# MAIN
# ============================================================
def main():
    global locked_id, cmd_vx, cmd_vz, cmd_yaw, is_airborne, TARGET_ALTITUDE [cite: 17]

    parser = argparse.ArgumentParser(description='Drone Tracker') [cite: 17]
    parser.add_argument('--fc', default='ardupilot')
    parser.add_argument('--connect', default='127.0.0.1:14550')
    parser.add_argument('--source', default='0')
    parser.add_argument('--no-fly', action='store_true')
    parser.add_argument('--altitude', type=float, default=10.0)
    parser.add_argument('--gcs-port', type=int, default=8080)
    args = parser.parse_args() [cite: 17]

    TARGET_ALTITUDE = args.altitude [cite: 17]

    # ----------------------------------------------------------
    # Connect to Flight Controller
    # ----------------------------------------------------------
    print(f"[TRACKER] Connecting to FC ({args.fc}) on {args.connect}...") [cite: 18]
    fc = FCInterface(args.fc, args.connect) [cite: 18]
    if not fc.connect() and not args.no_fly: sys.exit(1) [cite: 18]

    fc_thread = threading.Thread(target=fc_thread_fn, args=(fc, args.no_fly), daemon=True) [cite: 18]
    fc_thread.start() [cite: 18]

    # ----------------------------------------------------------
    # Browser GCS Server
    # ----------------------------------------------------------
    gcs = GCSServer(port=args.gcs_port, frame_w=FRAME_W, frame_h=FRAME_H) [cite: 18]
    gcs.start() [cite: 18]

    # ----------------------------------------------------------
    # Vision Pipeline Setup
    # ----------------------------------------------------------
    print("[TRACKER] Loading YOLO11n...") [cite: 19]
    model = YOLO(YOLO_MODEL) [cite: 19]
    pid   = TrackerPID() [cite: 19]

    src = int(args.source) if args.source.isdigit() else args.source [cite: 19]
    cap = cv2.VideoCapture(src) [cite: 19]
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W) [cite: 19]
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H) [cite: 19]

    if not cap.isOpened():
        print(f"[TRACKER] Cannot open source '{args.source}'. Using test pattern.") [cite: 19, 20]
        cap = None [cite: 20]

    fps, t_prev = 0.0, time.time() [cite: 20]
    prev_filter = 'any'  # <--- FIXED: Initialized here to completely avoid UnboundLocalError

    try:
        while not flag_quit.is_set(): [cite: 20]
            # ---- Frame acquisition ----
            if cap is not None: [cite: 20]
                ret, frame = cap.read() [cite: 20]
                if not ret: [cite: 21]
                    if isinstance(src, str): [cite: 21]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) [cite: 21]
                        continue [cite: 21]
                    break [cite: 21]
                frame = cv2.resize(frame, (FRAME_W, FRAME_H)) [cite: 21, 22]
            else:
                frame = np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8) [cite: 22]
                tx = int(FRAME_W/2 + 150*np.sin(time.time()*0.4)) [cite: 22]
                cv2.rectangle(frame, (tx-35, 120), (tx+35, 340), (0, 0, 200), -1) [cite: 22]

            t_now  = time.time() [cite: 23]
            fps    = 0.9*fps + 0.1*(1.0/max(t_now - t_prev, 1e-4)) [cite: 23]
            t_prev = t_now [cite: 23]

            # ---- Extract RC Telemetry Filters ----
            rc_lock, rc_release, rc_filter = fc.consume_rc_events() [cite: 23]
            active_classes = CLASS_GROUPS.get(rc_filter, CLASS_GROUPS['any']) [cite: 23]

            # ---- YOLO + ByteTrack ----
            results = model.track(frame, persist=True, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, classes=active_classes, tracker="bytetrack.yaml") [cite: 24]

            tracks = [] [cite: 24]
            if results[0].boxes.id is not None: [cite: 24]
                boxes   = results[0].boxes.xyxy.cpu().numpy() [cite: 24]
                ids     = results[0].boxes.id.cpu().numpy().astype(int) [cite: 24, 25]
                confs   = results[0].boxes.conf.cpu().numpy() [cite: 25]
                classes = results[0].boxes.cls.cpu().numpy().astype(int) [cite: 25]

                for i in range(len(ids)): [cite: 25]
                    tid = ids[i] [cite: 25]
                    entry = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], tid, confs[i], classes[i]] [cite: 25, 26]
                    tracks.append(entry) [cite: 26]
                    last_boxes[tid] = entry [cite: 26]

                active_ids = set(ids) [cite: 26]
                for tid in list(lost_frames.keys()): [cite: 26]
                    if tid not in active_ids: [cite: 27]
                        lost_frames[tid] += 1 [cite: 27]
                        if lost_frames[tid] > MAX_LOST_FRAMES: [cite: 27]
                            if tid == locked_id: [cite: 27]
                                locked_id = None [cite: 28]
                                pid.reset() [cite: 28]
                                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0 [cite: 28, 29]
                            lost_frames.pop(tid, None) [cite: 29]
                            last_boxes.pop(tid, None) [cite: 29]
                    else: lost_frames[tid] = 0 [cite: 29]

            # ---- Hardware RC Switch Commands ----
            if rc_filter != prev_filter: [cite: 30]
                print(f"\n[RC FILTER] CH9 changed target filter to: {rc_filter.upper()}") [cite: 30]
                prev_filter = rc_filter [cite: 30]
                
            if rc_lock: [cite: 30]
                if tracks: [cite: 31]
                    best = max(tracks, key=lambda t: (t[2]-t[0])*(t[3]-t[1])) [cite: 31]
                    locked_id = int(best[4]) [cite: 31]
                    pid.reset() [cite: 31]
                    print(f"\n[RC LOCK] Engaged ID:{locked_id} ({TARGET_CLASSES.get(int(best[6]), '?')}) via CH7") [cite: 31, 32]
                else:
                    print("\n[RC LOCK] CH7 toggled, but no targets visible to lock.") [cite: 32]
            
            if rc_release: [cite: 32]
                locked_id = None [cite: 32]
                pid.reset() [cite: 33]
                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0 [cite: 33]

            # ---- GCS Tap-to-Lock ----
            tap = gcs.consume_tap() [cite: 33]
            if tap is not None and tracks: [cite: 33]
                tx, ty = tap [cite: 33]
                nearest = min(tracks, key=lambda t: ((t[0]+t[2])/2 - tx)**2 + ((t[1]+t[3])/2 - ty)**2) [cite: 34]
                nx1, ny1, nx2, ny2 = int(nearest[0]), int(nearest[1]), int(nearest[2]), int(nearest[3]) [cite: 34]
                if nx1-30 <= tx <= nx2+30 and ny1-30 <= ty <= ny2+30: [cite: 34]
                    locked_id = int(nearest[4]) [cite: 34]
                    pid.reset() [cite: 35]

            # ---- PID update ----
            locked_box = None [cite: 35]
            if locked_id is not None: [cite: 35]
                locked_track = next((t for t in tracks if int(t[4]) == locked_id), None) [cite: 35]
                if locked_track is not None: [cite: 36]
                    locked_box = (int(locked_track[0]), int(locked_track[1]), int(locked_track[2]), int(locked_track[3])) [cite: 36]
                    vx, vz, yaw = pid.compute(locked_box, FRAME_W, FRAME_H) [cite: 36]
                    with cmd_lock: cmd_vx, cmd_vz, cmd_yaw = vx, vz, yaw [cite: 36]
                else: [cite: 37]
                    with cmd_lock:  [cite: 37]
                        cmd_vx *= 0.8 [cite: 37]
                        cmd_vz *= 0.8 [cite: 38]
                        cmd_yaw *= 0.8 [cite: 38]
            else:
                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0 [cite: 38]
                pid.reset() [cite: 38]

            # ---- Draw + display ----
            frame, locked_box = draw_tracks(frame, tracks, locked_id) [cite: 38]
            frame = draw_hud(frame, locked_id, locked_box, len(tracks), fps, pid, args.no_fly, rc_filter) [cite: 39]

            cv2.imshow("Drone Tracker", frame) [cite: 39]
            gcs.update_frame(frame) [cite: 39]

            # ---- Keyboard input ----
            key = cv2.waitKey(1) & 0xFF [cite: 39]
            if key in [ord('q'), ord('Q')]: flag_quit.set() [cite: 39]
            elif key in [ord('l'), ord('L')] and tracks: [cite: 40]
                best = max(tracks, key=lambda t: (t[2]-t[0])*(t[3]-t[1])) [cite: 40]
                locked_id = int(best[4]) [cite: 40]
                pid.reset() [cite: 40]
            elif key in [ord('r'), ord('R')]: [cite: 40]
                locked_id = None [cite: 40]
                pid.reset() [cite: 41]
                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0 [cite: 41]
            elif key in [ord('t'), ord('T')]: flag_takeoff.set() [cite: 41]
            elif key in [ord('h'), ord('H')]: flag_land.set() [cite: 41]
            elif key in [ord('+'), ord('=')]: pid.setpoint_area = max(0.02, pid.setpoint_area - 0.01) [cite: 41]
            elif key in [ord('-'), ord('_')]: pid.setpoint_area = min(0.50, pid.setpoint_area + 0.01) [cite: 42]

    finally:
        print("\n[TRACKER] Shutting down...") [cite: 42]
        flag_quit.set() [cite: 42]
        with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0 [cite: 42]
        if not args.no_fly and is_airborne: fc.land() [cite: 42]
        
        fc_thread.join(timeout=3) [cite: 42]
        if cap is not None: cap.release() [cite: 42]
        cv2.destroyAllWindows() [cite: 43]
        
        fc.disconnect() [cite: 43]
        print("[TRACKER] Stopped.") [cite: 43]

if __name__ == '__main__':
    main()
