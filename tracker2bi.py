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
from fc_interface5   import FCInterface
from gcs_server        import GCSServer

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_MODEL      = "yolo11n.pt"
FRAME_W         = 640
FRAME_H         = 480
CONF_THRESHOLD  = 0.45
IOU_THRESHOLD   = 0.45
MAX_LOST_FRAMES = 25
TARGET_ALTITUDE = 10.0

TARGET_CLASSES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# RC CH9 class filter groups
CLASS_GROUPS = {
    'any':     [0, 2, 3, 5, 7],
    'person':  [0],
    'vehicle': [2, 3, 5, 7],
}

CLASS_COLORS = {
    0: (0, 255, 0), 2: (255, 100, 0), 3: (0, 200, 255),
    5: (128, 0, 255), 7: (0, 80, 255),
}

# ============================================================
# GLOBAL STATE 
# ============================================================
locked_id   = None
lost_frames = defaultdict(int)
last_boxes = {}

cmd_vx, cmd_vz, cmd_yaw = 0.0, 0.0, 0.0
cmd_lock = threading.Lock()

telemetry  = {}
telem_lock = threading.Lock()

flag_takeoff = threading.Event()
flag_land    = threading.Event()
flag_quit    = threading.Event()
is_airborne  = False

# ============================================================
# FC COMMAND THREAD 
# ============================================================
def fc_thread_fn(fc, no_fly: bool):
    global is_airborne, cmd_vx, cmd_vz, cmd_yaw

    print("[FC THREAD] Started.")
    while not flag_quit.is_set():
        if flag_takeoff.is_set() and not is_airborne:
            flag_takeoff.clear()
            if not no_fly:
                try:
                    fc.arm_and_takeoff(TARGET_ALTITUDE)
                    is_airborne = True
                    print("[FC THREAD] Airborne.")
                except Exception as e: print(f"[FC THREAD] Takeoff failed: {e}")
            else: is_airborne = True

        if flag_land.is_set():
            flag_land.clear()
            if not no_fly: fc.land()
            is_airborne = False
            with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0
            print("[FC THREAD] Landing.")

        if is_airborne and not no_fly:
            with cmd_lock: vx, vz, yaw = cmd_vx, cmd_vz, cmd_yaw
            try: fc.send_velocity(vx, 0.0, vz, yaw)
            except Exception: pass

        try:
            t = fc.get_telemetry()
            with telem_lock: telemetry.update(t)
        except Exception: pass

        time.sleep(0.05)
    print("[FC THREAD] Stopped.")


# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_tracks(frame, tracks, locked_id):
    fh, fw  = frame.shape[:2]
    cx_f, cy_f = fw // 2, fh // 2
    locked_box = None

    cv2.drawMarker(frame, (cx_f, cy_f), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)

    for t in tracks:
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        tid, conf, cid = int(t[4]), float(t[5]), int(t[6])
        
        cls_name = TARGET_CLASSES.get(cid, "object")
        color    = CLASS_COLORS.get(cid, (200, 200, 200))
        is_lock  = (tid == locked_id)

        if is_lock:
            draw_col, thickness = (0, 255, 255), 3
            locked_box = (x1, y1, x2, y2)
        else:
            draw_col, thickness = color, 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_col, thickness)

        if is_lock:
            L = 18
            for (px, py, dx, dy) in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
                cv2.line(frame, (px, py), (px + dx*L, py), draw_col, 2)
                cv2.line(frame, (px, py), (px, py + dy*L), draw_col, 2)
            cv2.arrowedLine(frame, (cx_f, cy_f), ((x1 + x2) // 2, (y1 + y2) // 2), (0, 255, 255), 1, tipLength=0.12)

        label = f"{'[LOCK] ' if is_lock else ''}ID:{tid} {cls_name} {conf:.2f}"
        lbl_w, lbl_h = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.rectangle(frame, (x1, y1 - lbl_h - 6), (x1 + lbl_w + 4, y1), draw_col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(frame, ((x1 + x2) // 2, (y1 + y2) // 2), 4, draw_col, -1)

    return frame, locked_box

def draw_hud(frame, locked_id, locked_box, track_count, fps, pid, no_fly, rc_filter):
    fh, fw = frame.shape[:2]
    with cmd_lock: vx, vz, yaw = cmd_vx, cmd_vz, cmd_yaw
    with telem_lock: t = dict(telemetry)

    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (230, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    mode = t.get('mode', '?')
    ardupilot_mode_map = {"0": "STABILIZE", "4": "GUIDED", "5": "LOITER", "6": "RTL", "9": "LAND"}
    mode_text = ardupilot_mode_map.get(str(mode), f"MODE:{mode}")

    lines = [
        f"FPS    : {fps:.1f}", f"Tracks : {track_count}", f"Lock   : {locked_id if locked_id is not None else 'NONE'}",
        f"ALT    : {t.get('alt', 0.0):.1f}m", f"HDG    : {t.get('heading', 0)}deg", f"MODE   : {mode_text}",
        f"BAT    : {t.get('battery', 0.0):.1f}V", f"SATS   : {t.get('gps_sats', 0)}",
        f"FLTR   : {rc_filter.upper()}"
    ]

    if locked_box is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh)
        lines += [
            f"Vx     : {vx:+.2f} m/s", f"Vz     : {vz:+.2f} m/s", f"Yaw    : {yaw:+.2f} r/s",
            f"ErrX   : {dbg['error_x']:+.3f}", f"ErrY   : {dbg.get('error_y', 0):+.3f}",
            f"Area   : {dbg['current_area']:.3f}/{dbg['setpoint_area']:.3f}"
        ]

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    status, scol = ("GROUNDED", (0, 0, 255)) if not is_airborne else ("TRACKING", (0, 255, 0)) if locked_id is not None else ("SCANNING", (0, 165, 255))
    cv2.putText(frame, status, (fw - 140, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, scol, 2, cv2.LINE_AA)
    if no_fly: cv2.putText(frame, "[NO-FLY MODE]", (fw - 165, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 80, 255), 1, cv2.LINE_AA)

    if locked_box is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh)
        bx, bt, bb = fw - 28, 70, fh - 70
        fill = int(min(dbg['current_area'] / max(dbg['setpoint_area'] * 2, 0.001), 1.0) * (bb - bt))
        bcol = (0, 255, 0) if abs(dbg['error_area']) < 0.02 else (0, 140, 255)
        cv2.rectangle(frame, (bx, bt), (bx+14, bb), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, bb-fill), (bx+14, bb), bcol, -1)
        cv2.putText(frame, "DIST", (bx-4, bt-6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.putText(frame, "[L]Lock [R]Rel [T]Takeoff [H]Home [+/-]Dist [Q]Quit", (5, fh - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)
    return frame

# ============================================================
# MAIN
# ============================================================
def main():
    global locked_id, cmd_vx, cmd_vz, cmd_yaw, is_airborne, TARGET_ALTITUDE

    parser = argparse.ArgumentParser(description='Drone Tracker')
    parser.add_argument('--fc', default='ardupilot')
    parser.add_argument('--connect', default='127.0.0.1:14550')
    parser.add_argument('--source', default='0')
    parser.add_argument('--no-fly', action='store_true')
    parser.add_argument('--altitude', type=float, default=10.0)
    parser.add_argument('--gcs-port', type=int, default=8080)
    args = parser.parse_args()

    TARGET_ALTITUDE = args.altitude

    # ----------------------------------------------------------
    # Connect to Flight Controller
    # ----------------------------------------------------------
    print(f"[TRACKER] Connecting to FC ({args.fc}) on {args.connect}...")
    fc = FCInterface(args.fc, args.connect)
    if not fc.connect() and not args.no_fly: sys.exit(1)

    fc_thread = threading.Thread(target=fc_thread_fn, args=(fc, args.no_fly), daemon=True)
    fc_thread.start()

    # ----------------------------------------------------------
    # Browser GCS Server
    # ----------------------------------------------------------
    gcs = GCSServer(port=args.gcs_port, frame_w=FRAME_W, frame_h=FRAME_H)
    gcs.start()

    # ----------------------------------------------------------
    # Vision Pipeline Setup
    # ----------------------------------------------------------
    print("[TRACKER] Loading YOLO11n...")
    model = YOLO(YOLO_MODEL)
    pid   = TrackerPID()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"[TRACKER] Cannot open source '{args.source}'. Using test pattern.")
        cap = None

    fps, t_prev = 0.0, time.time()
    prev_filter = "any"

    try:
        while not flag_quit.is_set():
            # ---- Frame acquisition ----
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(src, str):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            else:
                frame = np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8)
                tx = int(FRAME_W/2 + 150*np.sin(time.time()*0.4))
                cv2.rectangle(frame, (tx-35, 120), (tx+35, 340), (0, 0, 200), -1)

            t_now  = time.time()
            fps    = 0.9*fps + 0.1*(1.0/max(t_now - t_prev, 1e-4))
            t_prev = t_now

            # ---- Extract RC Telemetry Filters ----
            rc_lock, rc_release, rc_filter = fc.consume_rc_events()
            active_classes = CLASS_GROUPS.get(rc_filter, CLASS_GROUPS['any'])

            # ---- YOLO + ByteTrack ----
            results = model.track(frame, persist=True, verbose=False, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, classes=active_classes, tracker="bytetrack.yaml")

            tracks = []
            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for i in range(len(ids)):
                    tid = ids[i]
                    entry = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], tid, confs[i], classes[i]]
                    tracks.append(entry)
                    last_boxes[tid] = entry

                active_ids = set(ids)
                for tid in list(lost_frames.keys()):
                    if tid not in active_ids:
                        lost_frames[tid] += 1
                        if lost_frames[tid] > MAX_LOST_FRAMES:
                            if tid == locked_id:
                                locked_id = None
                                pid.reset()
                                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0
                            lost_frames.pop(tid, None)
                            last_boxes.pop(tid, None)
                    else: lost_frames[tid] = 0

            # ---- Hardware RC Switch Commands ---- added rc_filter and updated lock and track statements
            if rc_filter != prev_filter:
                print(f"\n[RC FILTER] CH9 changed target filter to: {rc_filter.upper()}")
                prev_filter = rc_filter
                
            if rc_lock:
                if tracks:
                    best = max(tracks, key=lambda t: (t[2]-t[0])*(t[3]-t[1]))
                    locked_id = int(best[4])
                    pid.reset()
                    print(f"\n[RC LOCK] Engaged ID:{locked_id} ({TARGET_CLASSES.get(int(best[6]), '?')}) via CH7")
                else:
                    print("\n[RC LOCK] CH7 toggled, but no targets visible to lock.")
            
            if rc_release:
                locked_id = None
                pid.reset()
                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0

            # ---- GCS Tap-to-Lock ----
            tap = gcs.consume_tap()
            if tap is not None and tracks:
                tx, ty = tap
                nearest = min(tracks, key=lambda t: ((t[0]+t[2])/2 - tx)**2 + ((t[1]+t[3])/2 - ty)**2)
                nx1, ny1, nx2, ny2 = int(nearest[0]), int(nearest[1]), int(nearest[2]), int(nearest[3])
                if nx1-30 <= tx <= nx2+30 and ny1-30 <= ty <= ny2+30:
                    locked_id = int(nearest[4])
                    pid.reset()

            # ---- PID update ----
            locked_box = None
            if locked_id is not None:
                locked_track = next((t for t in tracks if int(t[4]) == locked_id), None)
                if locked_track is not None:
                    locked_box = (int(locked_track[0]), int(locked_track[1]), int(locked_track[2]), int(locked_track[3]))
                    vx, vz, yaw = pid.compute(locked_box, FRAME_W, FRAME_H)
                    with cmd_lock: cmd_vx, cmd_vz, cmd_yaw = vx, vz, yaw
                else:
                    with cmd_lock: cmd_vx *= 0.8; cmd_vz *= 0.8; cmd_yaw *= 0.8
            else:
                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0
                pid.reset()

            # ---- Draw + display ----
            frame, locked_box = draw_tracks(frame, tracks, locked_id)
            frame = draw_hud(frame, locked_id, locked_box, len(tracks), fps, pid, args.no_fly, rc_filter)

            cv2.imshow("Drone Tracker", frame)
            gcs.update_frame(frame)

            # ---- Keyboard input ----
            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q')]: flag_quit.set()
            elif key in [ord('l'), ord('L')] and tracks:
                best = max(tracks, key=lambda t: (t[2]-t[0])*(t[3]-t[1]))
                locked_id = int(best[4])
                pid.reset()
            elif key in [ord('r'), ord('R')]:
                locked_id = None
                pid.reset()
                with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0
            elif key in [ord('t'), ord('T')]: flag_takeoff.set()
            elif key in [ord('h'), ord('H')]: flag_land.set()
            elif key in [ord('+'), ord('=')]: pid.setpoint_area = max(0.02, pid.setpoint_area - 0.01)
            elif key in [ord('-'), ord('_')]: pid.setpoint_area = min(0.50, pid.setpoint_area + 0.01)

    finally:
        print("\n[TRACKER] Shutting down...")
        flag_quit.set()
        with cmd_lock: cmd_vx = cmd_vz = cmd_yaw = 0.0
        if not args.no_fly and is_airborne: fc.land()
        
        fc_thread.join(timeout=3)
        if cap is not None: cap.release()
        cv2.destroyAllWindows()
        
        # Explicitly release port bindings and connection handles 
        fc.disconnect()
        print("[TRACKER] Stopped.")

if __name__ == '__main__':
    main()
