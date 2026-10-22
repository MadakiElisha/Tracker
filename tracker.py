#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Production Tracker (v2.1)

 Perception (YOLO + ByteTrack) -> lock -> 3-axis PID -> FC.

 Production properties
   * every constant comes from config.yaml (--config to override)
   * no module-level mutable state: one TrackerState passed explicitly
   * safety supervisor runs in the FC thread, independent of the vision
     loop: video loss, link loss, battery, pilot override -> hover/RTL/release
   * the FC thread refuses commands older than the watchdog timeout, so a
     hung vision loop can never leave the drone flying on stale velocity
   * every frame is written to a CSV run log (analyze_run.py scores it)
   * errors are logged and counted, never swallowed

 Usage
   python3 tracker.py --connect 127.0.0.1:14550 --source clip.mp4 --no-fly
   python3 tracker.py --connect 127.0.0.1:14550 --source 0
   python3 tracker.py --connect /dev/ttyAMA0 --no-display      # on the drone

 Keys (video window): L lock-largest  R release  T takeoff
                      H RTL  K land  +/- follow distance  Q quit(+RTL)
=============================================================
"""

import argparse
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from tracker_logger  import setup_logging, get_logger
from tracker_state   import TrackerState
from pid_controller2 import TrackerPID
from profiles        import apply_profile
from class_vote      import ClassVote
from fc_interface5   import FCInterface
from gcs_server      import GCSServer
from mouse_lock      import MouseLocker
from safety          import SafetySupervisor, HOVER, RELEASE_CONTROL, RTL, limit_vertical
from run_logger      import RunLogger

WINDOW = "Drone Tracker v2.1"


# ============================================================
# CONFIG LOADER
# ============================================================
def load_config(path: str = "config.yaml") -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"[FATAL] config file not found: {cfg_path.resolve()}")
        sys.exit(1)
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ============================================================
# FC COMMAND + SAFETY THREAD (20 Hz, independent of vision loop)
# ============================================================
def fc_thread_fn(fc, state: TrackerState, cfg: dict, no_fly: bool,
                 supervisor: SafetySupervisor):
    log = get_logger("fc_thread")
    target_alt  = cfg['flight']['target_altitude']
    cmd_timeout = cfg['fc']['watchdog_timeout']
    last_action = "NONE"
    disarmed_since = None
    max_fwd = cfg['flight']['max_forward_speed']
    max_yaw = cfg['flight']['max_yaw_rate']
    warn_v  = cfg['safety'].get('battery_warn_v', 0.0)
    battery_warned = False
    log.info("FC/safety thread started (no_fly=%s).", no_fly)

    while not state.flag_quit.is_set():
        try:
            # -- takeoff request ------------------------------------------
            if state.flag_takeoff.is_set() and not state.is_airborne:
                state.flag_takeoff.clear()
                if no_fly or fc is None:
                    log.info("no-fly: simulating airborne.")
                    state.is_airborne = True
                else:
                    ok = False
                    try:
                        ok = fc.arm_and_takeoff(target_alt)
                    except Exception as e:
                        state.errors.record('takeoff', e, 40)
                    if ok:
                        state.is_airborne = True
                        state.rtl_issued  = False
                        disarmed_since    = None
                        log.info("Airborne at %.1f m.", target_alt)
                    else:
                        log.error("Takeoff failed.")
                        try:
                            if fc.get_telemetry().get('armed'):
                                log.error("Vehicle is armed after failed takeoff: landing.")
                                fc.land()
                        except Exception as e:
                            state.errors.record('takeoff_recovery', e, 40)

            # -- operator RTL / land --------------------------------------
            for flag, name in ((state.flag_rtl, "rtl"), (state.flag_land, "land")):
                if flag.is_set():
                    flag.clear()
                    state.cmd.zero()
                    if not no_fly and fc is not None:
                        try:
                            (fc.rtl if name == "rtl" else fc.land)()
                            state.rtl_issued = True
                        except Exception as e:
                            state.errors.record(name, e, 40)
                    else:
                        state.is_airborne = False

            if no_fly or fc is None:
                time.sleep(0.05)
                continue

            telem = fc.get_telemetry()
            state.telem.update(telem)

            if warn_v > 0 and 0 < telem.get('battery', 0.0) < warn_v:
                if not battery_warned:
                    log.warning("Battery low: %.2f V (warn %.2f V).", telem['battery'], warn_v)
                    battery_warned = True
            else:
                battery_warned = False

            # -- landed detection -----------------------------------------
            if state.is_airborne:
                if not telem.get('armed', False) and telem.get('alt', 0.0) < 0.5:
                    disarmed_since = disarmed_since or time.monotonic()
                    if time.monotonic() - disarmed_since > 2.0:
                        log.info("Landed / disarmed.")
                        state.is_airborne = False
                        state.rtl_issued  = False
                        state.cmd.zero()
                else:
                    disarmed_since = None

            # -- safety supervisor ----------------------------------------
            decision = supervisor.evaluate(
                airborne=state.is_airborne,
                takeoff_in_progress=False,
                mode=telem.get('mode', 'UNKNOWN'),
                telemetry_age_s=fc.telemetry_age(),
                battery_v=telem.get('battery', 0.0))
            state.safety = decision
            if decision.action != last_action:
                if decision.action == "NONE":
                    log.info("Safety cleared (was %s).", last_action)
                else:
                    log.warning("SAFETY %s: %s", decision.action, decision.reason)
                last_action = decision.action

            if decision.action == HOVER:
                state.cmd.zero()
                fc.hover()
            elif decision.action == RELEASE_CONTROL:
                state.cmd.zero()
                if not decision.reason.startswith("pilot_override"):
                    fc.hover()
            elif decision.action == RTL:
                state.cmd.zero()
                if not state.rtl_issued:
                    fc.rtl()
                    state.rtl_issued = True

            # -- velocity stream ------------------------------------------
            if state.is_airborne and not decision.suppress_commands:
                vx, vz, yaw, age = state.cmd.read_with_age()
                if age > cmd_timeout:
                    vx = vz = yaw = 0.0        # vision loop stale: hold position
                vx  = max(-max_fwd, min(max_fwd, vx))       # master speed caps,
                yaw = max(-max_yaw, min(max_yaw, yaw))      # independent of PID tuning
                vz = limit_vertical(vz, telem.get('alt', 0.0),
                                    cfg['flight']['min_altitude'],
                                    cfg['flight']['max_altitude'])
                fc.send_velocity(vx, 0.0, vz, yaw)

        except Exception as e:
            state.errors.record('fc_thread', e, 40)

        time.sleep(0.05)

    log.info("FC/safety thread stopped.")


# ============================================================
# DRAWING HELPERS
# All receive data explicitly — no global access.
# ============================================================

def draw_tracks(frame: np.ndarray,
                tracks: list,
                locked_id,
                class_colors: dict,
                target_classes: dict,
                reid_warn: float = 0.60) -> tuple:
    """
    Draw bounding boxes, labels, and lock indicator.
    Returns (annotated_frame, locked_box_or_None).
    """
    fh, fw  = frame.shape[:2]
    cx_f    = fw // 2
    cy_f    = fh // 2
    locked_box = None

    # Frame centre crosshair
    cv2.drawMarker(frame, (cx_f, cy_f),
                   (255, 255, 255), cv2.MARKER_CROSS, 24, 1)

    for t in tracks:
        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
        tid      = int(t[4])
        conf     = float(t[5])
        cid      = int(t[6])
        is_lock  = (tid == locked_id)
        cls_name = target_classes.get(cid, "object")
        color    = class_colors.get(cid, [200, 200, 200])
        bgr      = tuple(color) if isinstance(color, list) else color

        if is_lock:
            draw_col   = (0, 255, 255)   # cyan
            thickness  = 3
            locked_box = (x1, y1, x2, y2)
        else:
            draw_col  = bgr
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

        # Re-ID confidence warning
        reid_warn_txt = " [RE-ID?]" if (is_lock and conf < reid_warn) else ""
        label = (f"{'[LOCK] ' if is_lock else ''}"
                 f"ID:{tid} {cls_name} {conf:.2f}{reid_warn_txt}")

        lbl_w, lbl_h = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0]
        cv2.rectangle(frame,
                      (x1, y1 - lbl_h - 6),
                      (x1 + lbl_w + 4, y1),
                      draw_col, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 0, 0), 1, cv2.LINE_AA)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, draw_col, -1)

        if is_lock:
            cv2.arrowedLine(frame, (cx_f, cy_f), (cx, cy),
                            (0, 255, 255), 1, tipLength=0.12)

    return frame, locked_box


def draw_hud(frame: np.ndarray,
             state: TrackerState,
             tracks: list,
             locked_box,
             fps: float,
             pid: TrackerPID,
             rc_filter: str,
             no_fly: bool,
             cfg: dict) -> np.ndarray:
    """
    Draw the telemetry HUD overlay.
    Receives all data explicitly — no global access.
    """
    fh, fw = frame.shape[:2]
    vx, vz, yaw = state.cmd.read()
    telem = state.telem.snapshot()

    # Determine locked track confidence
    locked_conf, locked_cls = None, None
    if state.locked_id is not None:
        for t in tracks:
            if int(t[4]) == state.locked_id:
                locked_conf = float(t[5])
                locked_cls  = int(t[6])
                break

    # Decode ArduPilot mode number to name
    mode_map = {
        "0": "STABILIZE", "2": "ALT_HOLD", "3": "AUTO",
        "4": "GUIDED",    "5": "LOITER",   "6": "RTL",
        "7": "CIRCLE",    "9": "LAND",     "16": "POSHOLD",
    }
    mode_str = mode_map.get(
        str(telem.get('mode', '')), telem.get('mode', '?'))

    # Semi-transparent panel background
    panel_h = 290 if locked_box is not None else 210
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (245, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # HUD lines
    lines = [
        f"FPS    : {fps:.1f}",
        f"Tracks : {len(tracks)}",
        f"Lock   : {state.locked_id if state.locked_id is not None else 'NONE'}",
        f"ALT    : {telem.get('alt', 0.0):.1f}m",
        f"HDG    : {telem.get('heading', 0)}deg",
        f"MODE   : {mode_str}",
        f"BAT    : {telem.get('battery', 0.0):.1f}V",
        f"SATS   : {telem.get('gps_sats', 0)}",
        f"FLTR   : {rc_filter.upper()}",
        f"ERRS   : {state.errors.total()}",
    ]
    sd = state.safety
    if sd is not None and sd.action != "NONE":
        lines.append(f"SAFE   : {sd.action}")

    if locked_box is not None and locked_conf is not None:
        dbg = pid.get_debug_info(locked_box, fw, fh, class_id=locked_cls)
        lines += [
            f"ReID   : {locked_conf:.2f} conf",
            f"LostFr : {state.lost_frames.get(state.locked_id, 0)}/"
                      f"{cfg['tracking']['max_lost_frames']}",
            f"Vx     : {vx:+.2f} m/s",
            f"Vz     : {vz:+.2f} m/s",
            f"Yaw    : {yaw:+.2f} r/s",
            f"ErrX   : {dbg['error_x']:+.3f}",
            f"ErrY   : {dbg.get('error_y', 0):+.3f}",
            f"Area   : {dbg['current_area']:.3f}/{dbg['setpoint_area']:.3f}",
        ]

    for i, line in enumerate(lines):
        color = (0, 80, 255) if (line.startswith("SAFE") or (
                line.startswith("ERRS") and state.errors.total() > 0)) \
                else (0, 255, 0)
        cv2.putText(frame, line, (10, 22 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    color, 1, cv2.LINE_AA)

    # Status badge
    if not state.is_airborne:
        status, scol = "GROUNDED",  (0, 0, 255)
    elif state.locked_id is not None:
        status, scol = "TRACKING",  (0, 255, 0)
    else:
        status, scol = "SCANNING",  (0, 165, 255)

    cv2.putText(frame, status, (fw - 145, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, scol, 2, cv2.LINE_AA)

    if no_fly:
        cv2.putText(frame, "[NO-FLY]", (fw - 110, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (0, 80, 255), 1, cv2.LINE_AA)

    # Distance bar
    if locked_box is not None:
        dbg   = pid.get_debug_info(locked_box, fw, fh, class_id=locked_cls)
        bx    = fw - 28
        bt, bb = 70, fh - 70
        bh    = bb - bt
        fill  = int(min(dbg['current_area'] /
                        max(dbg['setpoint_area'] * 2, 0.001), 1.0) * bh)
        bcol  = ((0, 255, 0)
                 if abs(dbg['error_dist']) < pid.pitch_pid.deadzone
                 else (0, 140, 255))
        cv2.rectangle(frame, (bx, bt), (bx+14, bb), (50, 50, 50), -1)
        cv2.rectangle(frame, (bx, bb-fill), (bx+14, bb), bcol, -1)
        cv2.putText(frame, "DIST", (bx-4, bt-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                    (200, 200, 200), 1)

    # Footer
    footer = "[L]Lock [R]Rel [T]TKO [H]RTL [K]Land [+/-]Dist [Q]Quit"
    cv2.putText(frame, footer, (5, fh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                (160, 160, 160), 1, cv2.LINE_AA)

    return frame



# ============================================================
# MAIN
# ============================================================
def main():
    # -- config (path first, so every other default can come from it) ----
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='config.yaml')
    known, _ = pre.parse_known_args()
    cfg = load_config(known.config)

    setup_logging(cfg.get('logging', {}))
    log = get_logger("tracker")
    log.info("Drone Tracker v2.1 starting (config: %s).", known.config)

    parser = argparse.ArgumentParser(description='Drone Tracker v2.1', parents=[pre])
    parser.add_argument('--fc',      default=cfg['fc'].get('type', 'ardupilot'))
    parser.add_argument('--connect', default=cfg['fc'].get('connection', '127.0.0.1:14550'))
    parser.add_argument('--source',  default=str(cfg['video'].get('source', '0')),
                        help='Camera index or video file path')
    parser.add_argument('--profile', default=cfg.get('active_profile'),
                        help='Mission profile from config.yaml `profiles:` '
                             f"({', '.join(sorted((cfg.get('profiles') or {})))})")
    parser.add_argument('--no-fly',     action='store_true',
                        help='Perception + PID only; no FC commands are sent')
    parser.add_argument('--no-display', action='store_true',
                        help='Headless: no OpenCV window')
    args = parser.parse_args()
    if args.profile:
        try:
            cfg = apply_profile(cfg, args.profile)
        except ValueError as e:
            log.error("%s", e)
            sys.exit(2)
        log.info("Mission profile '%s': cap %.1f m/s, altitude %.1f m, gimbal=%s",
                 args.profile, cfg['flight']['max_forward_speed'], cfg['flight']['target_altitude'],
                 cfg.get('camera', {}).get('gimbal', False))
        if cfg['flight']['max_forward_speed'] > 10.0 and not cfg.get('camera', {}).get('gimbal', False):
            log.warning("Profile '%s' flies above ~10 m/s without a stabilised camera: tilt "
                        "corrections have little margin left; a gimbal is recommended.", args.profile)

    fw, fh = cfg['video']['frame_width'], cfg['video']['frame_height']
    state = TrackerState()
    supervisor = SafetySupervisor(cfg['safety'])

    # -- flight controller ----------------------------------------------
    if args.no_fly:      # don't stall start-up for 15 s when there is no FC
        cfg['fc']['connect_timeout_s'] = min(cfg['fc'].get('connect_timeout_s', 15.0), 3.0)
    fc = FCInterface(args.fc, args.connect, cfg['fc'], cfg['rc'], cfg['safety'])
    log.info("Connecting to FC (%s) on %s...", args.fc, args.connect)
    try:
        connected = fc.connect()
    except Exception as e:
        log.error("FC connection error: %s", e)
        connected = False
    if not connected:
        if not args.no_fly:
            log.error("FC connection failed; refusing to start without --no-fly.")
            sys.exit(1)
        log.warning("FC unavailable; continuing in --no-fly mode without FC.")
        fc = None

    fc_t = threading.Thread(target=fc_thread_fn, name="fc-thread", daemon=True,
                            args=(fc, state, cfg, args.no_fly, supervisor))
    fc_t.start()

    # -- GCS browser (waitress) -------------------------------------------
    gcs_cfg = cfg['gcs']
    gcs = GCSServer(port=gcs_cfg['port'], frame_w=fw, frame_h=fh,
                    jpeg_quality=gcs_cfg.get('jpeg_quality', 70),
                    fps_cap=gcs_cfg.get('stream_fps_cap', 25),
                    token=gcs_cfg.get('token', ''),
                    host=gcs_cfg.get('host', '0.0.0.0'))
    if not gcs_cfg.get('token'):
        log.warning("GCS has no token: anyone on this network can lock targets "
                    "(set gcs.token in config.yaml before flying).")

    def _serve_gcs():
        try:
            import waitress
            log.info("GCS on http://%s:%d (waitress)", gcs.host, gcs.port)
            waitress.serve(gcs.app, host=gcs.host, port=gcs.port,
                           threads=4, _quiet=True)
        except ImportError:
            log.warning("waitress missing (pip install waitress); using Flask dev server.")
            gcs.start()
        except Exception as e:
            state.errors.record('gcs', e, 40)
    threading.Thread(target=_serve_gcs, name="gcs", daemon=True).start()

    mouse_locker = MouseLocker()

    # -- model ------------------------------------------------------------
    from ultralytics import YOLO
    model_path = cfg['model']['path']
    if not Path(model_path).exists():
        log.error("Model file not found: %s  (run: python3 model_download.py)", model_path)
        sys.exit(1)
    model = YOLO(model_path)
    log.info("Model loaded: %s", model_path)

    cam = cfg.get('camera', {})
    pid_cfg = dict(cfg['pid'])
    pid_cfg.setdefault('camera_hfov_deg', cam.get('hfov_deg', 90.0))
    pid_cfg.setdefault('camera_pitch_deg', cam.get('pitch_deg', 20.0))
    gimbal = bool(cam.get('gimbal', False))
    pid = TrackerPID.from_config(pid_cfg)
    class_vote = ClassVote(cfg['tracking'].get('class_vote_frames', 60))

    target_classes = {int(k): v for k, v in cfg['classes']['targets'].items()}
    class_groups   = cfg['classes']['groups']
    class_colors   = {int(k): v for k, v in cfg['classes']['colors'].items()}
    max_lost       = cfg['tracking']['max_lost_frames']
    half_life      = cfg['tracking'].get('lost_decay_half_life_s', 0.15)

    run_cfg = cfg.get('run_log', {})
    run_log = RunLogger(run_cfg.get('dir', 'logs/runs'), run_cfg.get('enabled', True))
    if run_log.path:
        log.info("Run log: %s", run_log.path)

    # -- video ------------------------------------------------------------
    src = int(args.source) if str(args.source).isdigit() else args.source
    if isinstance(src, str) and src.startswith('gz:'):
        try:
            from gz_camera import GzCamera
            cap = GzCamera(src[3:] or '/drone/camera')
        except ImportError as e:
            log.error("gz: source needs gz-transport13/gz-msgs10 Python bindings: %s", e)
            sys.exit(2)
    else:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  fw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)
    if not cap.isOpened():
        log.warning("Cannot open source '%s'; using synthetic test pattern.", args.source)
        cap = None

    fps, t_prev = 0.0, time.monotonic()
    frame_n = 0
    prev_rc_filter, rc_filter = 'any', 'any'
    display_ready, just_rewound = False, False
    last_stale_log = last_stale_row = 0.0

    log.info("Running. Keys: L lock, R release, T takeoff, H RTL, K land, Q quit.")

    try:
        while not state.flag_quit.is_set():
            rc_lock_evt = rc_rel_evt = False

            # ---- frame acquisition ---------------------------------------
            got_frame = True
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(src, str) and not just_rewound:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    # loop a clip
                        just_rewound = True
                        continue
                    got_frame = False
                else:
                    just_rewound = False
                    frame = cv2.resize(frame, (fw, fh))
            else:
                frame = np.full((fh, fw, 3), 60, dtype=np.uint8)
                tx = int(fw / 2 + 150 * np.sin(time.time() * 0.4))
                cv2.rectangle(frame, (tx - 35, 120), (tx + 35, 340), (0, 0, 200), -1)

            now = time.monotonic()

            if not got_frame:
                # Video lost: never keep commanding on stale vision.
                state.cmd.zero()
                if now - last_stale_log > 1.0:
                    log.warning("No video frame (source '%s').", args.source)
                    last_stale_log = now
                if now - last_stale_row > 0.1:
                    sd = state.safety
                    run_log.row(frame=frame_n, airborne=int(state.is_airborne),
                                mode=state.telem.snapshot().get('mode', ''),
                                tracks=0, locked_id=state.locked_id if state.locked_id is not None else "",
                                target_visible=0, cmd_vx=0, cmd_vz=0, cmd_yaw=0,
                                safety_action=sd.action if sd else "NONE",
                                safety_reason=sd.reason if sd else "",
                                errors=state.errors.total())
                    last_stale_row = now
                if not args.no_display and (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q')):
                    state.flag_quit.set()
                time.sleep(0.02)
                continue

            frame_n += 1
            supervisor.note_frame()
            dt = max(now - t_prev, 1e-4)
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
            t_prev = now

            # ---- RC events -----------------------------------------------
            if fc is not None:
                try:
                    rc_lock_evt, rc_rel_evt, rc_filter = fc.consume_rc_events()
                    if rc_lock_evt: log.info("RC CH%d: lock event.", cfg['rc']['lock_channel'])
                    if rc_rel_evt:  log.info("RC CH%d: release event.", cfg['rc']['release_channel'])
                    if rc_filter != prev_rc_filter:
                        log.info("RC CH%d: class filter -> %s", cfg['rc']['filter_channel'], rc_filter)
                        prev_rc_filter = rc_filter
                except Exception as e:
                    state.errors.record('rc_events', e)
                    rc_lock_evt = rc_rel_evt = False

            # ---- detection + tracking ------------------------------------
            active_classes = class_groups.get(rc_filter, class_groups['any'])
            try:
                results = model.track(
                    frame, persist=True, verbose=False,
                    conf=cfg['model']['conf_threshold'],
                    iou=cfg['model']['iou_threshold'],
                    classes=active_classes,
                    tracker=cfg['tracking']['tracker_config'])
            except Exception as e:
                state.errors.record('inference', e)
                results = None

            tracks = []
            if results and results[0].boxes.id is not None:
                boxes   = results[0].boxes.xyxy.cpu().numpy()
                ids     = results[0].boxes.id.cpu().numpy().astype(int)
                confs   = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                for i in range(len(ids)):
                    entry = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                             ids[i], confs[i], classes[i]]
                    tracks.append(entry)
                    state.last_boxes[int(ids[i])] = entry

            # Every frame, including empty ones, so a vanished target is counted.
            state.update_lost_frames(active_ids={int(t[4]) for t in tracks},
                                     max_lost=max_lost, pid=pid)

            # ---- safety gate -----------------------------------------------
            decision   = state.safety
            suppressed = decision is not None and decision.suppress_commands
            if suppressed and state.locked_id is not None:
                state.release_lock(f"SAFETY:{decision.reason}", pid)

            def try_lock(track, source):
                if suppressed:
                    log.warning("Lock request (%s) ignored: safety %s active.",
                                source, decision.action)
                    return
                state.lock_target(int(track[4]),
                                  target_classes.get(int(track[6]), '?'), source, pid)

            def largest():
                return max(tracks, key=lambda t: (t[2] - t[0]) * (t[3] - t[1]))

            # ---- target assignment -----------------------------------------
            if rc_lock_evt:
                if tracks: try_lock(largest(), "RC_LOCK")
                else:      log.info("RC lock: no targets visible.")
            if rc_rel_evt:
                state.release_lock("RC_RELEASE", pid)

            click = mouse_locker.consume_click()
            if click:
                hit = state.resolve_tap(click, tracks)
                if hit is not None: try_lock(hit, f"MOUSE{click}")
                else: log.debug("Mouse click %s hit no target.", click)

            tap = gcs.consume_tap()
            if tap:
                hit = state.resolve_tap(tap, tracks)
                if hit is not None: try_lock(hit, f"GCS_TAP{tap}")
                else: log.debug("GCS tap %s hit no target.", tap)

            # ---- control ---------------------------------------------------
            locked_box, locked_cls = None, None
            if state.locked_id is not None and not suppressed:
                lt = next((t for t in tracks if int(t[4]) == state.locked_id), None)
                if lt is not None:
                    locked_box = (int(lt[0]), int(lt[1]), int(lt[2]), int(lt[3]))
                    locked_cls = class_vote.update(state.locked_id, int(lt[6]))
                    tm = fc.get_telemetry() if fc is not None else {}
                    hdg = math.radians(tm.get('heading', 0))
                    v_fwd = tm.get('vx', 0.0) * math.cos(hdg) + tm.get('vy', 0.0) * math.sin(hdg)
                    vx, vz, yaw = pid.compute(
                        locked_box, fw, fh, class_id=locked_cls,
                        tilt_rad=0.0 if gimbal else tm.get('tilt_rad', 0.0),
                        alt_m=tm.get('alt') if fc is not None else None,
                        v_drone_fwd=v_fwd)
                    state.cmd.update(vx, vz, yaw)
                else:
                    state.cmd.decay(0.5 ** (dt / half_life))   # time-based decay
            else:
                state.cmd.zero()
                pid.reset()

            # ---- draw ------------------------------------------------------
            frame, drawn_box = draw_tracks(frame, tracks, state.locked_id,
                                           class_colors, target_classes,
                                           cfg['tracking'].get('reid_warn_threshold', 0.60))
            frame = draw_hud(frame, state, tracks, drawn_box, fps, pid,
                             rc_filter, args.no_fly, cfg)

            # ---- run log -----------------------------------------------------
            telem = state.telem.snapshot()
            cvx, cvz, cyaw = state.cmd.read()
            dbg = pid.get_debug_info(locked_box, fw, fh, class_id=locked_cls) \
                if locked_box is not None else {}
            run_log.row(
                frame=frame_n, fps=round(fps, 1), airborne=int(state.is_airborne),
                mode=telem.get('mode', ''), alt=round(telem.get('alt', 0.0), 2),
                battery=round(telem.get('battery', 0.0), 2),
                tracks=len(tracks),
                locked_id=state.locked_id if state.locked_id is not None else "",
                target_visible=int(locked_box is not None),
                lost_frames=state.lost_frames.get(state.locked_id, 0) if state.locked_id is not None else "",
                err_x=dbg.get('error_x', ""), err_y=dbg.get('error_y', ""),
                area=dbg.get('current_area', ""), setpoint=dbg.get('setpoint_area', ""),
                cmd_vx=round(cvx, 3), cmd_vz=round(cvz, 3), cmd_yaw=round(cyaw, 3),
                safety_action=decision.action if decision else "NONE",
                safety_reason=decision.reason if decision else "",
                errors=state.errors.total())

            # ---- display + keys ----------------------------------------------
            gcs.update_frame(frame)
            key = 0xFF
            if not args.no_display:
                cv2.imshow(WINDOW, frame)
                if not display_ready:
                    cv2.setMouseCallback(WINDOW, mouse_locker.handle_click)
                    display_ready = True
                key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q')):
                state.flag_quit.set()
            elif key in (ord('l'), ord('L')):
                if tracks: try_lock(largest(), "KBD")
                else: log.info("KBD lock: no targets visible.")
            elif key in (ord('r'), ord('R')):
                state.release_lock("KBD", pid)
            elif key in (ord('t'), ord('T')):
                log.info("KBD: takeoff requested.")
                state.flag_takeoff.set()
            elif key in (ord('h'), ord('H')):
                log.info("KBD: RTL requested.")
                state.flag_rtl.set()
            elif key in (ord('k'), ord('K')):
                log.info("KBD: LAND requested.")
                state.flag_land.set()
            elif key in (ord('+'), ord('=')):
                pid.setpoint_area = max(0.02, pid.setpoint_area - 0.01)
                pid.setpoint_by_class = {k: max(0.02, v - 0.01) for k, v in pid.setpoint_by_class.items()}
                log.info("Follow distance increased (default setpoint %.2f).", pid.setpoint_area)
            elif key in (ord('-'), ord('_')):
                pid.setpoint_area = min(0.50, pid.setpoint_area + 0.01)
                pid.setpoint_by_class = {k: min(0.50, v + 0.01) for k, v in pid.setpoint_by_class.items()}
                log.info("Follow distance decreased (default setpoint %.2f).", pid.setpoint_area)

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        log.info("Shutdown initiated.")
        state.flag_quit.set()
        state.cmd.zero()
        if fc is not None:
            fc.abort_takeoff()
            if not args.no_fly and state.is_airborne and not state.rtl_issued:
                log.warning("Quitting while airborne: commanding RTL.")
                try:
                    fc.rtl()
                except Exception as e:
                    log.error("RTL on shutdown failed: %s", e)
        fc_t.join(timeout=3)
        if cap is not None:
            cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        run_log.close()
        if fc is not None:
            fc.disconnect()
        log.info("Stopped. Tracker errors: %d %s",
                 state.errors.total(), state.errors.summary())


if __name__ == '__main__':
    main()
