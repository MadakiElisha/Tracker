#!/usr/bin/env python3
"""
Measure detection + tracking speed and output on the target board, no flight controller needed.

    python3 scripts/hailo_selftest.py                         # csi:0 camera, model chosen like tracker.py
    python3 scripts/hailo_selftest.py --source clip.mp4 --frames 300
    python3 scripts/hailo_selftest.py --model visdrone_yolo11n.pt   # compare against CPU PyTorch

Prints the backend, end-to-end loop rate, detector time (mean / p95), and what it detected per
VisDrone class. PASS needs >= --min-fps; the tracker's profiles were tuned at 19 fps in Gazebo.
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np   # noqa: E402
import yaml          # noqa: E402

from model_select import resolve_model, apply_camera_preset   # noqa: E402


def open_source(spec, w, h, cfg):
    if spec == 'csi' or str(spec).startswith('csi:'):
        from csi_camera import CsiCamera, parse_csi_spec
        return CsiCamera(parse_csi_spec(spec), w, h, cfg['video'].get('fps', 30), cfg.get('camera', {}).get('csi'))
    import cv2
    cap = cv2.VideoCapture(int(spec) if str(spec).isdigit() else spec)
    return cap


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default=str(ROOT / 'config.yaml'))
    ap.add_argument('--source', default='csi:0')
    ap.add_argument('--model', default=None)
    ap.add_argument('--camera', default=None, help='camera preset (config.yaml camera.sensors)')
    ap.add_argument('--frames', type=int, default=200)
    ap.add_argument('--min-fps', type=float, default=19.0)
    ap.add_argument('--conf', type=float, default=None, help='override model.conf_threshold (e.g. 0.1 to see weak detections)')
    ap.add_argument('--save', default=None, help='write an annotated JPEG of the last frame here')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.camera:
        cfg = apply_camera_preset(cfg, args.camera)
    w, h = cfg['video']['frame_width'], cfg['video']['frame_height']
    path, backend = resolve_model(cfg['model'], args.model)
    from ultralytics import YOLO
    model = YOLO(path, task='detect') if backend == 'hailo' else YOLO(path)
    names = model.names
    print(f"[selftest] model {path} | backend {backend} | source {args.source}")

    cap = open_source(args.source, w, h, cfg)
    if not cap.isOpened():
        print(f"[selftest] FAIL cannot open source {args.source}")
        return 1

    det_ms, seen, n, t0, last = [], collections.Counter(), 0, None, None
    ids = set()
    print(f"[selftest] camera open; first frame loads the model (can take 10-30 s on the CPU), "
          f"then {args.frames} frames, progress every 25", flush=True)
    last_ok = time.monotonic()
    while n < args.frames:
        ok, frame = cap.read()
        if not ok:
            if not getattr(cap, 'is_live', False):
                if n == 0:
                    print("[selftest] FAIL no frames")
                    return 1
                break
            if time.monotonic() - last_ok > 5.0:
                print("[selftest] FAIL camera stopped delivering frames for 5 s "
                      "(another program using the camera? check: rpicam-hello --list-cameras)")
                return 1
            continue
        last_ok = time.monotonic()
        if frame.shape[1] != w or frame.shape[0] != h:
            import cv2
            frame = cv2.resize(frame, (w, h))
        t = time.monotonic()
        r = model.track(frame, persist=True, verbose=False, conf=args.conf if args.conf is not None else cfg['model']['conf_threshold'],
                        iou=cfg['model']['iou_threshold'], tracker=cfg['tracking']['tracker_config'])
        det_ms.append((time.monotonic() - t) * 1000.0)
        if t0 is None:
            t0 = time.monotonic()          # first frame includes model warm-up; not counted
            det_ms.clear()
            print(f"[selftest] model ready ({(t0 - t) :.1f} s warm-up); measuring...", flush=True)
        else:
            n += 1
            if n % 25 == 0:
                el = time.monotonic() - t0
                print(f"[selftest]   {n}/{args.frames} frames, {n / el:.1f} fps so far, "
                      f"seen: {dict(seen) or 'nothing yet'}", flush=True)
        b = r[0].boxes
        for c in b.cls.cpu().numpy().astype(int):
            seen[names.get(int(c), str(c))] += 1
        if b.id is not None:
            ids.update(int(i) for i in b.id.cpu().numpy())
        last = r[0]
    elapsed = time.monotonic() - t0 if t0 else 0.0
    try:
        cap.release()
    except Exception:
        pass
    if n == 0 or elapsed <= 0:
        print("[selftest] FAIL no frames processed")
        return 1
    fps = n / elapsed
    d = np.array(det_ms)
    print(f"[selftest] frames {n} | loop {fps:.1f} fps | detect+track {d.mean():.1f} ms mean, "
          f"{np.percentile(d, 95):.1f} ms p95")
    print(f"[selftest] detections per class: {dict(seen) or 'none'} | distinct track ids: {len(ids)}")
    if args.save and last is not None:
        import cv2
        cv2.imwrite(args.save, last.plot())
        print(f"[selftest] annotated frame: {args.save}")
    verdict = fps >= args.min_fps
    print(f"[selftest] {'PASS' if verdict else 'FAIL'} (need >= {args.min_fps:.0f} fps)")
    return 0 if verdict else 1


if __name__ == '__main__':
    sys.exit(main())
