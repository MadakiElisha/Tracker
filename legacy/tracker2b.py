#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Integrated 3D Tracker (Phase 4 + Phase 5)

 Wires together:
   YOLO11n + ByteTrack     → perception
   pid_controller2.py      → 3-axis PID (yaw, pitch, Z)
   fc_interface3dv.py      → ArduPilot / PX4 HAL
   rc_monitor.py           → RC switch target assignment
   gcs_server.py           → Browser tap-to-lock

 Usage:
   python3 tracker.py --fc ardupilot --connect 127.0.0.1:14550
   python3 tracker.py --fc ardupilot --connect 127.0.0.1:14550 --source clip.mp4
   python3 tracker.py --fc ardupilot --connect 127.0.0.1:14550 --no-fly

 Keyboard (video window must be focused):
   L / R  — lock largest / release
   T / H  — takeoff / return home
   + / -  — increase / decrease follow distance
   Q      — quit and RTL
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
from fc_interface3dv   import FCInterface
from rc_monitor        import RCMonitor
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

TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# RC CH9 class filter groups
CLASS_GROUPS = {
    'any':     [0, 2, 3, 5, 7],
    'person':  [0],
    'vehicle': [2, 3, 5, 7],
}

CLASS_COLORS = {
    0: (0,   255,   0),
    2: (255, 100,   0),
    3: (0,   200, 255),
    5: (128,   0, 255),
    7: (0,    80, 255),
}

# ============================================================
# GLOBAL STATE  (shared between main thread and FC thread)
# ============================================================
locked_id   = None
lost_frames = defaultdict(int)
last_boxes  = {}

cmd_vx   = 0.0
cmd_vz   = 0.0
cmd_yaw  = 0.0
cmd_lock = threading.Lock()

telemetry  = {}
telem_lock = threading.Lock()

flag_takeoff = threading.Event()
flag_land    = threading.Event()
flag_quit    = threading.Event()
is_airborne  = False

# ============================================================
# FC COMMAND THREAD  — 20 Hz, decoupled from vision loop
# ============================================================
def fc_thread_fn(fc, no_fly: bool):
    global is_airborne, cmd_vx, cmd_vz, cmd_yaw

    print("[FC THREAD] Started.")

    while not flag_quit.is_set():

        # Takeoff
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
                print("[FC THREAD] --no-fly active, simulating airborne.")
                is_airborne = True

        # Land
        if flag_land.is_set():
            flag_land.clear()
            if not no_fly:
                fc.land()
            is_airborne = False
            with cmd_lock:
                cmd_vx = cmd_vz = cmd_yaw = 0.0
            print("[FC THREAD] Landing.")

        # Send velocity
        if is_airborne:
            with cmd_lock:
                vx, vz, yaw = cmd_vx, cmd_vz, cmd_yaw
            if not no_fly:
                try:
                    fc.send_velocity(vx, 0.0, vz, yaw)
                except Exception as e:
                    print(f"[FC THREAD] send_velocity error: {e}")

        # Read telemetry
        try:
            t = fc.get_telemetry()
            with telem_lock:
                telemetry.update(t)
        except Exception:
            pass

        time.sleep(0.05)   # 20 Hz

    print("[FC THREAD] Stopped.")


# ============================================================
# DRAWING HELPERS
# ============================================================
def draw_tracks(frame, tracks, locked_id):
    fh, fw  = frame.shape[:2]
    cx_f    = fw // 2
    cy_f    = fh // 2
    locked_box = None

    # Frame-centre crosshair
    cv2.drawMarker(frame, (cx_f, cy_f), (255, 255, 255),
                   cv2.MARKER_CROSS, 24, 1)

    for t in tracks:
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        tid      = int(t[4])
        conf     = float(t[5])
        cid      = int(t[6])
        cls_name = TARGET_CLASSES.get(cid, "object")
        color    = CLASS_COLORS.get(cid, (200, 200, 200))
        is_lock  = (tid == locked_id)

        if is_lock:
            draw_col   = (0, 255, 255)
            thickness  = 3
            locked_box = (x1, y1, x2, y2)
        else:
            draw_col  = color
            thickness = 2

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_col, thickness)

        # Corner accents on locked target
        if is_lock:
            L = 18
            for (px, py, dx, dy) in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1)
            ]:
                cv2.line(frame, (px, py), (px + dx*L, py), draw_col, 2)
                cv2.line(frame, (px, py), (px, py + dy*L), draw_col, 2)

        # Label
        label   = f"{'[LOCK] ' if is_lock else ''}ID:{tid} {cls_name} {conf:.2f}"
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

        # Error arrow from frame centre to locked target
        if is_lock:
            cv2.arrowedLine(frame, (cx_f, cy_f), (cx, cy),
                            (0, 255, 255), 1, tipLength=0.12)

    return frame, locked_box


def draw_hud(frame, locked_id, locked_box, track_count,
             fps, pid, no_fly):
    fh, fw = frame.shape[:2]

    with cmd_lock:
        vx, vz, yaw = cmd_vx, cmd_vz, cmd_yaw
    with telem_lock:
        t = dict(telemetry)

    # Left telemetry panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (230, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    lines = [
        f"FPS    : {fps:.1f}",
        f"Tracks : {track_count}",
        f"Lock   : {locked_id if locked_id is not None else 'NONE'}",
        f"ALT    : {t.get('alt', 0.0):.1f}m",
        f"HDG    : {t.get('heading', 0)}deg",
        f"MODE   : {t.get('mode', '?')}",
        f"BAT    : {t.get('battery', 0.0):.1f}V",
        f"SATS   : {t.get('gps_sats', 0)}",
    ]

    if locked_box is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh)
        lines += [
            f"Vx     : {vx:+.2f} m/s",
            f"Vz     : {vz:+.2f} m/s",
            f"Yaw    : {yaw:+.2f} r/s",
            f"ErrX   : {dbg['error_x']:+.3f}",
            f"ErrY   : {dbg.get('error_y', 0):+.3f}",
            f"Area   : {dbg['current_area']:.3f}/{dbg['setpoint_area']:.3f}",
        ]

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # Status badge
    if not is_airborne:
        status, scol = "GROUNDED",  (0, 0, 255)
    elif locked_id is not None:
        status, scol = "TRACKING",  (0, 255, 0)
    else:
        status, scol = "SCANNING",  (0, 165, 255)

    cv2.putText(frame, status, (fw - 140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, scol, 2, cv2.LINE_AA)

    if no_fly:
        cv2.putText(frame, "[NO-FLY MODE]", (fw - 165, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 80, 255), 1, cv2.LINE_AA)

    # Distance bar (right edge)
    if locked_box is not None:
        dbg     = pid.get_debug_info(locked_box, fw, fh)
        bx      = fw - 28
        bt, bb  = 70, fh - 70
        bh      = bb - bt
        fill    = int(min(dbg['current_area'] /
                          max(dbg['setpoint_area'] * 2, 0.001), 1.0) * bh)
        bcol    = ((0, 255, 0)
                   if abs(dbg['error_area']) < 0.02
                   else (0, 140, 255))
        cv2.rectangle(frame, (bx, bt), (bx+14, bb), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, bb-fill), (bx+14, bb), bcol, -1)
        cv2.putText(frame, "DIST", (bx-4, bt-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 200), 1)

    # Footer controls
    cv2.putText(frame,
                "[L]Lock [R]Release [T]Takeoff [H]Home [+/-]Dist [Q]Quit",
                (5, fh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# ============================================================
# MAIN
# ============================================================
def main():
    global locked_id, cmd_vx, cmd_vz, cmd_yaw, is_airborne, TARGET_ALTITUDE

    parser = argparse.ArgumentParser(description='Drone Tracker')
    parser.add_argument('--fc',       default='ardupilot',
                        help='FC type: ardupilot | px4')
    parser.add_argument('--connect',  default='127.0.0.1:14550',
                        help='FC connection string')
    parser.add_argument('--source',   default='0',
                        help='Camera index (0) or video file path')
    parser.add_argument('--no-fly',   action='store_true',
                        help='Perception + PID only — no FC commands sent')
    parser.add_argument('--altitude', type=float, default=10.0,
                        help='Takeoff altitude in metres')
    parser.add_argument('--gcs-port', type=int, default=8080,
                        help='Browser GCS server port')
    args = parser.parse_args()

    TARGET_ALTITUDE = args.altitude

    # ----------------------------------------------------------
    # Connect to flight controller
    # ----------------------------------------------------------
    print(f"[TRACKER] Connecting to FC ({args.fc}) on {args.connect}...")
    fc = FCInterface(args.fc, args.connect)
    try:
        fc.connect()
    except Exception as e:
        print(f"[TRACKER] FC connection failed: {e}")
        if not args.no_fly:
            sys.exit(1)
        print("[TRACKER] Continuing in --no-fly mode.")

    # Start FC command thread
    fc_thread = threading.Thread(
        target=fc_thread_fn, args=(fc, args.no_fly), daemon=True)
    fc_thread.start()

    # ----------------------------------------------------------
    # Phase 5A: RC Monitor
    # ----------------------------------------------------------
    rc_monitor = None
    if hasattr(fc.backend, 'master'):
        rc_monitor = RCMonitor(fc.backend.master)
        rc_monitor.start()
        print("[TRACKER] RC monitor started (CH7=lock, CH8=release, CH9=filter).")
    else:
        print("[TRACKER] RC monitor unavailable — backend has no 'master'.")

    # ----------------------------------------------------------
    # Phase 5B: Browser GCS tap-to-lock server
    # ----------------------------------------------------------
    gcs = GCSServer(port=args.gcs_port, frame_w=FRAME_W, frame_h=FRAME_H)
    gcs.start()

    # ----------------------------------------------------------
    # Load YOLO and open video source
    # ----------------------------------------------------------
    print("[TRACKER] Loading YOLO11n...")
    model = YOLO(YOLO_MODEL)
    pid   = TrackerPID()

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        print(f"[TRACKER] Cannot open source '{args.source}'. "
              f"Using test pattern.")
        cap = None

    print("[TRACKER] Running.")
    print("  L = lock  R = release  T = takeoff  H = home  Q = quit")
    print(f"  GCS browser: http://<WSL2-IP>:{args.gcs_port}")

    fps    = 0.0
    t_prev = time.time()

    try:
        while not flag_quit.is_set():

            # ---- Frame acquisition ----
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(src, str):           # loop video file
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break
                frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            else:
                frame = np.full((FRAME_H, FRAME_W, 3), 60, dtype=np.uint8)
                tx = int(FRAME_W/2 + 150*np.sin(time.time()*0.4))
                cv2.rectangle(frame,
                              (tx-35, 120), (tx+35, 340),
                              (0, 0, 200), -1)

            # ---- FPS ----
            t_now  = time.time()
            fps    = 0.9*fps + 0.1*(1.0/max(t_now - t_prev, 1e-4))
            t_prev = t_now

            # ---- YOLO + ByteTrack ----
            # Apply RC CH9 class filter
            active_classes = CLASS_GROUPS['any']
            if rc_monitor is not None:
                active_classes = CLASS_GROUPS.get(
                    rc_monitor.get_class_filter(), CLASS_GROUPS['any'])

            results = model.track(
                frame,
                persist  = True,
                verbose  = False,
                conf     = CONF_THRESHOLD,
                iou      = IOU_THRESHOLD,
                classes  = active_classes,
                tracker  = "bytetrack.yaml",
            )

            # ---- Parse detections ----
            tracks = []
            if results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for i in range(len(ids)):
                    tid   = ids[i]
                    entry = [boxes[i][0], boxes[i][1],
                             boxes[i][2], boxes[i][3],
                             tid, confs[i], classes[i]]
                    tracks.append(entry)
                    last_boxes[tid] = entry

                active_ids = set(ids)
                for tid in list(lost_frames.keys()):
                    if tid not in active_ids:
                        lost_frames[tid] += 1
                        if lost_frames[tid] > MAX_LOST_FRAMES:
                            if tid == locked_id:
                                print(f"\n[TRACKER] Lock lost: "
                                      f"ID {tid} gone {MAX_LOST_FRAMES} frames.")
                                locked_id = None
                                pid.reset()
                                with cmd_lock:
                                    cmd_vx = cmd_vz = cmd_yaw = 0.0
                            lost_frames.pop(tid, None)
                            last_boxes.pop(tid, None)
                    else:
                        lost_frames[tid] = 0

            # ---- Phase 5A: RC switch events (CH7 lock / CH8 release) ----
            if rc_monitor is not None and rc_monitor.consume_lock_event():
                if tracks:
                    best      = max(tracks,
                                   key=lambda t: (t[2]-t[0])*(t[3]-t[1]))
                    locked_id = int(best[4])
                    pid.reset()
                    print(f"\n[RC LOCK] ID:{locked_id} "
                          f"({TARGET_CLASSES.get(int(best[6]), '?')})")
                else:
                    print("\n[RC LOCK] No targets visible.")

            if rc_monitor is not None and rc_monitor.consume_release_event():
                print(f"\n[RC RELEASE] Released ID:{locked_id}")
                locked_id = None
                pid.reset()
                with cmd_lock:
                    cmd_vx = cmd_vz = cmd_yaw = 0.0

            # ---- Phase 5B: GCS tap-to-lock ----
            tap = gcs.consume_tap()
            if tap is not None and tracks:
                tx, ty = tap
                nearest = min(tracks,
                              key=lambda t:
                                  ((t[0]+t[2])/2 - tx)**2 +
                                  ((t[1]+t[3])/2 - ty)**2)
                nx1, ny1, nx2, ny2 = (int(nearest[0]), int(nearest[1]),
                                      int(nearest[2]), int(nearest[3]))
                if nx1-30 <= tx <= nx2+30 and ny1-30 <= ty <= ny2+30:
                    locked_id = int(nearest[4])
                    pid.reset()
                    print(f"\n[GCS LOCK] ID:{locked_id} "
                          f"({TARGET_CLASSES.get(int(nearest[6]),'?')}) "
                          f"via tap ({tx},{ty})")
                else:
                    print(f"\n[GCS TAP] No target near ({tx},{ty}).")

            # ---- PID update ----
            locked_box = None
            if locked_id is not None:
                locked_track = next(
                    (t for t in tracks if int(t[4]) == locked_id), None)
                if locked_track is not None:
                    locked_box = (int(locked_track[0]), int(locked_track[1]),
                                  int(locked_track[2]), int(locked_track[3]))
                    vx, vz, yaw = pid.compute(locked_box, FRAME_W, FRAME_H)
                    with cmd_lock:
                        cmd_vx, cmd_vz, cmd_yaw = vx, vz, yaw
                else:
                    # Target temporarily invisible — coast and decay
                    with cmd_lock:
                        cmd_vx  *= 0.8
                        cmd_vz  *= 0.8
                        cmd_yaw *= 0.8
            else:
                with cmd_lock:
                    cmd_vx = cmd_vz = cmd_yaw = 0.0
                pid.reset()

            # ---- Draw + display ----
            frame, locked_box = draw_tracks(frame, tracks, locked_id)
            frame = draw_hud(frame, locked_id, locked_box,
                             len(tracks), fps, pid, args.no_fly)

            cv2.imshow("Drone Tracker", frame)
            gcs.update_frame(frame)   # send to browser — once per loop

            # ---- Keyboard input ----
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q')]:
                flag_quit.set()

            elif key in [ord('l'), ord('L')]:
                if tracks:
                    best      = max(tracks,
                                   key=lambda t: (t[2]-t[0])*(t[3]-t[1]))
                    locked_id = int(best[4])
                    pid.reset()
                    print(f"\n[LOCK] ID:{locked_id} "
                          f"({TARGET_CLASSES.get(int(best[6]),'?')})")
                else:
                    print("\n[LOCK] No targets visible.")

            elif key in [ord('r'), ord('R')]:
                print(f"\n[RELEASE] Was ID:{locked_id}")
                locked_id = None
                pid.reset()
                with cmd_lock:
                    cmd_vx = cmd_vz = cmd_yaw = 0.0

            elif key in [ord('t'), ord('T')]:
                flag_takeoff.set()

            elif key in [ord('h'), ord('H')]:
                flag_land.set()

            elif key in [ord('+'), ord('=')]:
                pid.setpoint_area = max(0.02, pid.setpoint_area - 0.01)
                print(f"\n[PID] Setpoint area → {pid.setpoint_area:.2f}")

            elif key in [ord('-'), ord('_')]:
                pid.setpoint_area = min(0.50, pid.setpoint_area + 0.01)
                print(f"\n[PID] Setpoint area → {pid.setpoint_area:.2f}")

    finally:
        print("\n[TRACKER] Shutting down...")
        flag_quit.set()
        with cmd_lock:
            cmd_vx = cmd_vz = cmd_yaw = 0.0
        if not args.no_fly and is_airborne:
            fc.land()
        fc_thread.join(timeout=3)
        if rc_monitor is not None:
            rc_monitor.stop()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        fc.disconnect()
        print("[TRACKER] Stopped.")


if __name__ == '__main__':
    main()
