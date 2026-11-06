#!/usr/bin/env python3
"""What does the detector actually see in the simulator camera? Class and confidence per detection, for N seconds.

    python3 scripts/probe_detections.py --seconds 20          # sim running, drone hovering with the target in view

Prints, per class, how many frames it appeared in and its confidence range, plus how often the largest vehicle
box changed class. Use it to tell 'detector does not see the car' from 'controller misbehaves'.
"""
import argparse
import os
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--topic", default="/drone/camera")
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    from ultralytics import YOLO
    from gz_camera import GzCamera
    names = {int(k): v for k, v in cfg["classes"]["targets"].items()}
    model = YOLO(cfg["model"]["path"])
    cap = GzCamera(a.topic)
    seen, conf, frames, flips, last_cls = Counter(), defaultdict(list), 0, 0, None
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        frames += 1
        r = model.predict(frame, verbose=False, conf=cfg["model"].get("conf_threshold", 0.35))[0]
        best = None
        for b in r.boxes:
            c = int(b.cls[0])
            if c in names:
                seen[c] += 1
                conf[c].append(float(b.conf[0]))
                area = float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                if c in (3, 4, 5, 8) and (best is None or area > best[0]):
                    best = (area, c)
        if best:
            flips += last_cls is not None and best[1] != last_cls
            last_cls = best[1]
    print(f"{frames} frames in {a.seconds:.0f} s")
    for c, n in seen.most_common():
        cs = sorted(conf[c])
        print(f"  {names[c]:<11} in {n:4d} detections ({n / max(frames, 1):.2f}/frame)  conf {cs[0]:.2f}-{cs[-1]:.2f}  median {cs[len(cs) // 2]:.2f}")
    if not seen:
        print("  nothing detected: is the target in view? is the confidence threshold too high?")
    print(f"  largest-vehicle class changed {flips} times (each change can move the follow-distance setpoint)")


if __name__ == "__main__":
    main()
