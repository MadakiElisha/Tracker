#!/usr/bin/env python3
"""

 Drone Tracker — Phase 2: Perception Pipeline
 
 Detects and tracks multiple object classes using:
   - YOLO11n for detection
   - ByteTrack for multi-object tracking
   - Kalman filter (built into ByteTrack) for occlusion
 
 Target classes: person, car, motorcycle, bus, truck
 Target locking: keyboard for now (RC/GCS in Phase 5)
 
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from collections import defaultdict

# Ultralytics YOLO + ByteTrack via boxmot
from ultralytics import YOLO

# ......
# CONFIGURATION
# ......
YOLO_MODEL      = "yolo11n.pt"
FRAME_W         = 640
FRAME_H         = 480
CONF_THRESHOLD  = 0.45        # min detection confidence
IOU_THRESHOLD   = 0.45        # NMS IOU threshold

# Classes we want to detect and track
TARGET_CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Colours per class (BGR)
CLASS_COLORS = {
    0: (0,   255,  0  ),   # person    → green
    2: (255, 100,  0  ),   # car       → blue-orange
    3: (0,   200,  255),   # motorcycle→ yellow
    5: (128, 0,    255),   # bus       → purple
    7: (0,   80,   255),   # truck     → red
}

# How many frames a track can be lost before we drop it
MAX_LOST_FRAMES = 20

# .....
# TRACKER STATE
# .....
locked_id       = None          # track ID we are following
lost_frames     = defaultdict(int)
last_boxes      = {}            # last known box per track ID

# .....
# HELPER — Draw everything on frame
# .....
def draw_detections(frame, tracks, locked_id):
    """
    Draw bounding boxes, track IDs, class labels and 
    lock indicator on the frame.
    
    tracks: list of [x1,y1,x2,y2, track_id, conf, class_id]
    """
    fh, fw = frame.shape[:2]
    cx_frame, cy_frame = fw // 2, fh // 2

    # Frame centre crosshair
    cv2.drawMarker(frame, (cx_frame, cy_frame),
                   (255, 255, 255), cv2.MARKER_CROSS, 20, 1)

    locked_box = None

    for track in tracks:
        x1, y1, x2, y2 = int(track[0]), int(track[1]), \
                          int(track[2]), int(track[3])
        tid   = int(track[4])
        conf  = float(track[5])
        cid   = int(track[6])

        color     = CLASS_COLORS.get(cid, (200, 200, 200))
        cls_name  = TARGET_CLASSES.get(cid, "object")
        is_locked = (tid == locked_id)

        # Locked target gets thicker box + different colour
        if is_locked:
            draw_color     = (0, 255, 255)   # cyan
            thickness      = 3
            locked_box     = (x1, y1, x2, y2)
        else:
            draw_color     = color
            thickness      = 2

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, thickness)

        # Label background
        label    = f"{'[LOCKED] ' if is_locked else ''}ID:{tid} {cls_name} {conf:.2f}"
        lbl_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame,
                      (x1, y1 - lbl_size[1] - 6),
                      (x1 + lbl_size[0] + 4, y1),
                      draw_color, -1)
        cv2.putText(frame, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

        # Centre dot
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, draw_color, -1)

        # Arrow from frame centre to locked target
        if is_locked:
            cv2.arrowedLine(frame,
                            (cx_frame, cy_frame), (cx, cy),
                            (0, 255, 255), 1, tipLength=0.15)

    return frame, locked_box


def draw_hud(frame, locked_id, locked_box, track_count, fps,
             frame_w, frame_h):
    """
    Draw the telemetry HUD overlay on the frame.
    """
    fh, fw = frame.shape[:2]

    # Semi-transparent panel background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (280, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    lines = [
        f"FPS:     {fps:.1f}",
        f"Tracks:  {track_count}",
        f"Lock ID: {locked_id if locked_id is not None else 'NONE'}",
    ]

    if locked_box is not None:
        x1, y1, x2, y2 = locked_box
        bw = x2 - x1
        bh = y2 - y1
        area_pct = (bw * bh) / (frame_w * frame_h) * 100
        cx = (x1 + x2) // 2
        err_x = (cx / frame_w) - 0.5
        lines += [
            f"Area:    {area_pct:.1f}%",
            f"Err X:   {err_x:+.3f}",
        ]

    status = "TRACKING" if locked_id is not None else "SCANNING"
    status_color = (0, 255, 0) if locked_id is not None else (0, 165, 255)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 1, cv2.LINE_AA)

    # Status badge
    cv2.putText(frame, status, (fw - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                status_color, 2, cv2.LINE_AA)

    # Controls reminder
    controls = "[L] Lock largest  [R] Release  [Q] Quit"
    cv2.putText(frame, controls, (5, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# .....
# MAIN
# .....
def main():
    global locked_id

    print("[PERCEPTION] Loading YOLO11n model...")
    model = YOLO(YOLO_MODEL)
    print(f"[PERCEPTION] Model loaded. Tracking classes: {TARGET_CLASSES}")

    # Use webcam 0 — in Phase 6 this becomes the Gazebo camera feed
    cap = cv2.VideoCapture('track_test_clip.mp4') # Temporal test with some video feed
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("  In simulation, use: cv2.VideoCapture('path/to/video')")
        print("  or connect a webcam.")
        # For sim testing without a camera, create a test pattern
        print("[INFO] Running in TEST PATTERN mode (no camera).")
        cap = None

    print("[PERCEPTION] Running. Controls:")
    print("  L = lock on largest visible target")
    print("  R = release lock")
    print("  Q = quit")

    fps      = 0.0
    t_prev   = time.time()
    frame_n  = 0

    # For test pattern mode
    test_x = 320

    while True:
        # --- Get frame ---
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Frame read failed.")
                time.sleep(0.05)
                #continue
                break # Exit loop when video ends
            
            # ADD THIS LINE: Force the frame to 640x480 so YOLO doesn't choke
            frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        else:
            # Synthetic test frame — moving red box on grey background
            frame = np.full((FRAME_H, FRAME_W, 3), 80, dtype=np.uint8)
            test_x = int(FRAME_W/2 + 100 * np.sin(time.time() * 0.5))
            cv2.rectangle(frame,
                          (test_x - 40, 100),
                          (test_x + 40, 300),
                          (0, 0, 200), -1)

        frame_n += 1

        # --- FPS calculation ---
        t_now = time.time()
        dt    = t_now - t_prev
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t_prev = t_now

        # --- Run YOLO tracking ---
        # persist=True keeps ByteTrack state across frames — critical
        # classes= filters to only our target classes
        print(f"[DEBUG] Processing frame {frame_n}...", end='\r')
        results = model.track(
            frame,
            persist     = True,
            verbose     = False,
            conf        = CONF_THRESHOLD,
            iou         = IOU_THRESHOLD,
            classes     = list(TARGET_CLASSES.keys()),
            tracker     = "bytetrack.yaml",
        )

        # --- Parse tracks ---
        tracks = []
        if results[0].boxes.id is not None:
            boxes   = results[0].boxes.xyxy.cpu().numpy()
            ids     = results[0].boxes.id.cpu().numpy().astype(int)
            confs   = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for i in range(len(ids)):
                tid = ids[i]
                tracks.append([
                    boxes[i][0], boxes[i][1],
                    boxes[i][2], boxes[i][3],
                    tid, confs[i], classes[i]
                ])
                last_boxes[tid] = tracks[-1]

            # Update lost frame counters
            active_ids = set(ids)
            for tid in list(lost_frames.keys()):
                if tid not in active_ids:
                    lost_frames[tid] += 1
                    if lost_frames[tid] > MAX_LOST_FRAMES:
                        # Permanently lost — if this was our lock, release
                        if tid == locked_id:
                            print(f"[TRACKER] Lock lost: ID {tid} gone "
                                  f"for {MAX_LOST_FRAMES} frames.")
                            locked_id = None
                        del lost_frames[tid]
                        last_boxes.pop(tid, None)
                else:
                    lost_frames[tid] = 0

        # --- Draw ---
        frame, locked_box = draw_detections(frame, tracks, locked_id)
        frame = draw_hud(frame, locked_id, locked_box,
                         len(tracks), fps, FRAME_W, FRAME_H)

        # --- Show ---
        #cv2.imshow("Drone Tracker — Phase 2 Perception", frame)
        # This is to test that it is using the feeds
        cv2.imshow("debug_test.mp4", frame)

        # --- Keyboard controls ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('l'):
            # Lock on the largest visible target by bounding box area
            if tracks:
                best    = max(tracks,
                              key=lambda t: (t[2]-t[0]) * (t[3]-t[1]))
                locked_id = int(best[4])
                cls_name  = TARGET_CLASSES.get(int(best[6]), "object")
                print(f"[LOCK] Locked on ID:{locked_id} ({cls_name})")
            else:
                print("[LOCK] No targets visible to lock on.")

        elif key == ord('r'):
            print(f"[LOCK] Released lock (was ID:{locked_id})")
            locked_id = None

    # Cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[PERCEPTION] Stopped.")


if __name__ == '__main__':
    main()
