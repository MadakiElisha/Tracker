#!/usr/bin/env python3
"""
Find the lens-motor value that gives the sharpest picture at flight distance (IMX519).

    python3 scripts/focus_sweep.py                 # coarse sweep 0..4000, then a fine pass
    python3 scripts/focus_sweep.py --save          # also keep one JPEG per value in focus_sweep/

Point the camera at a detailed scene MORE THAN 10 m away and keep it still. The camera is
started first and the lens is moved while it streams, exactly as tracker.py does it (starting
the camera resets the lens motor, so values set before start are lost). Prints value, sharpness
(variance of the Laplacian over the centre of the frame; higher = sharper) and brightness, then
the value to put in config.yaml camera.csi.vcm_focus.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from csi_camera import set_vcm_focus   # noqa: E402


def sharpness(bgr) -> tuple[float, float]:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    c = g[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    return float(cv2.Laplacian(c, cv2.CV_64F).var()), float(c.mean())


def measure(cam, value, settle, frames, save_dir):
    set_vcm_focus(value)
    time.sleep(settle)
    for _ in range(2):                      # drop frames exposed while the lens was moving
        cam.capture_array("main")
    scores, lums = [], []
    for _ in range(frames):
        f = cam.capture_array("main")[:, :, :3]
        s, l = sharpness(f)
        scores.append(s)
        lums.append(l)
    if save_dir:
        cv2.imwrite(str(save_dir / f"focus_{value:04d}.jpg"), f)
    return float(np.median(scores)), float(np.mean(lums))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--settle', type=float, default=0.4, help='seconds to wait after moving the lens')
    ap.add_argument('--frames', type=int, default=3, help='frames averaged per value')
    ap.add_argument('--save', action='store_true', help='keep one JPEG per value in focus_sweep/')
    ap.add_argument('--sensor', default='2328x1748', help='sensor mode WxH (full view; "" = libcamera picks)')
    args = ap.parse_args()

    from picamera2 import Picamera2
    cam = Picamera2(args.camera)
    kw = {}
    if args.sensor:
        w, h = (int(x) for x in args.sensor.lower().split('x'))
        kw['sensor'] = {"output_size": (w, h)}
    cam.configure(cam.create_still_configuration(main={"size": (1280, 960), "format": "RGB888"}, **kw))
    cam.start()
    print(f"[focus] sensor mode {args.sensor or 'auto'}")
    time.sleep(1.0)                         # let exposure settle
    save_dir = None
    if args.save:
        save_dir = ROOT / 'focus_sweep'
        save_dir.mkdir(exist_ok=True)

    try:
        if set_vcm_focus(0) is None:
            print("[focus] no lens motor found (no v4l-subdev with focus_absolute)")
            return 1
        results = {}
        print(f"{'value':>6} {'sharpness':>10} {'brightness':>11}")
        coarse = [0, 150, 300, 450, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2400, 2800, 3200, 3600, 4000]
        for v in coarse:
            results[v] = measure(cam, v, args.settle, args.frames, save_dir)
            print(f"{v:>6} {results[v][0]:>10.1f} {results[v][1]:>11.0f}")
        best = max(results, key=lambda k: results[k][0])
        lo, hi = max(0, best - 200), min(4095, best + 200)
        print(f"-- fine pass {lo}..{hi}")
        for v in range(lo, hi + 1, 40):
            if v in results:
                continue
            results[v] = measure(cam, v, args.settle, args.frames, save_dir)
            print(f"{v:>6} {results[v][0]:>10.1f} {results[v][1]:>11.0f}")
        best = max(results, key=lambda k: results[k][0])
        top, lum = results[best]
        floor = float(np.median([r[0] for r in results.values()]))
        ratio = top / max(floor, 1e-6)
        print(f"\n[focus] best value {best}: sharpness {top:.1f} ({ratio:.0f}x the out-of-focus level), "
              f"brightness {lum:.0f}/255")
        # Absolute sharpness depends on the scene (sky, plain walls score low); a clear peak is
        # what matters: in focus should be several times sharper than the defocused floor.
        if lum < 40:
            print("[focus] WARNING: scene too dark to judge focus; repeat in daylight")
        elif ratio < 4:
            print("[focus] WARNING: no clear peak. Is the scene detailed and >10 m away? "
                  "Does the lens actually move (listen/look while it runs)?")
        else:
            print(f"[focus] set in config.yaml:  camera.csi.vcm_focus: {best}")
        set_vcm_focus(best)
    finally:
        cam.stop()
        cam.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
