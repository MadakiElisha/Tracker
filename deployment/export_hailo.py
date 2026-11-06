#!/usr/bin/env python3
"""
=============================================================
 Export the VisDrone YOLO11n model to a Hailo HEF (Hailo-8)

 Runs on the dev PC (Linux x86_64) with the Hailo Dataflow Compiler (DFC)
 v3.x installed in the same Python environment (scripts/setup_hailo_dfc.sh).
 Ultralytics drives the whole chain: .pt -> ONNX -> Hailo parse -> INT8
 calibration on real VisDrone images -> HEF with on-chip NMS config.

 Output: visdrone_yolo11n_hailo_model/   (folder: .hef + metadata.yaml)
 Deploy: copy the whole folder next to tracker.py on the Pi; tracker.py picks it
         automatically when /dev/hailo0 exists (config.yaml model.backend: auto).

   python3 deployment/export_hailo.py                       # Hailo-8, VisDrone calibration
   python3 deployment/export_hailo.py --arch hailo8l        # 13 TOPS HAT instead
   python3 deployment/export_hailo.py --data ~/datasets/VisDrone/VisDrone.yaml

 The AI HAT+ 26 TOPS is a Hailo-8. A HEF compiled for one chip does not run
 on the other. The DFC version must match the HailoRT version on the Pi.
=============================================================
"""
from __future__ import annotations

import argparse
import platform
import shutil
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default='visdrone_yolo11n.pt', help='VisDrone .pt (python3 model_download.py)')
    ap.add_argument('--arch', default='hailo8', choices=['hailo8', 'hailo8l'],
                    help='Target chip (AI HAT+ 26 TOPS = hailo8)')
    ap.add_argument('--data', default='VisDrone.yaml',
                    help='Dataset YAML for INT8 calibration (Ultralytics downloads VisDrone.yaml, ~2 GB)')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.10,
                    help='On-chip NMS score floor. Keep it low: ByteTrack uses low-score boxes and '
                         'tracker.py applies model.conf_threshold itself.')
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='Fraction of the calibration split to use (lower = faster, less accurate)')
    args = ap.parse_args()

    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        print("[ERROR] Hailo compilation needs Linux x86_64 (run this on the dev PC, not the Pi).")
        return 1
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics missing in this environment (scripts/setup_hailo_dfc.sh installs it).")
        return 1
    try:
        import hailo_sdk_client  # noqa: F401
    except ImportError:
        print("[ERROR] Hailo Dataflow Compiler not installed in this environment.\n"
              "        Download the DFC v3.x wheel (Linux x86_64) from the Hailo Developer Zone, then:\n"
              "        bash scripts/setup_hailo_dfc.sh /path/to/hailo_dataflow_compiler-*.whl")
        return 1
    if not Path(args.weights).exists():
        print(f"[ERROR] {args.weights} not found (python3 model_download.py)")
        return 1

    print(f"[OK] Ultralytics {ultralytics.__version__}; target {args.arch}; calibration {args.data}")
    print("[INFO] Compiling takes 20-60 min (calibration + compile). Do not interrupt.")
    t0 = time.time()
    model = YOLO(args.weights)
    names = getattr(model, 'names', {})
    if names and 'pedestrian' not in [str(v) for v in names.values()]:
        print(f"[WARN] {args.weights} does not look like the VisDrone model (classes: {names}).")
    out = model.export(format='hailo', name=args.arch, data=args.data, imgsz=args.imgsz,
                       conf=args.conf, iou=args.iou, fraction=args.fraction)
    out_dir = Path(out)
    want = Path(f"{Path(args.weights).stem}_hailo_model")
    if out_dir.resolve() != want.resolve():
        if want.exists():
            shutil.rmtree(want)
        shutil.move(str(out_dir), str(want))
        out_dir = want
    hefs = list(out_dir.rglob('*.hef'))
    print(f"[DONE] {out_dir}/ in {(time.time() - t0) / 60:.1f} min: "
          f"{', '.join(f'{h.name} ({h.stat().st_size / 1e6:.1f} MB)' for h in hefs)}")
    print(f"[NEXT] scp -r {out_dir} <user>@<pi-host>:~/Tracker/   then on the Pi:\n"
          f"       python3 scripts/hailo_selftest.py --model {out_dir.name}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
