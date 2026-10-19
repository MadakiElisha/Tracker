#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — VisDrone Model Downloader

 Downloads the dronefreak/visdrone-yolov11n model from
 HuggingFace and verifies it loads correctly.

 NOTE: The model card at huggingface.co/dronefreak/visdrone-yolov11n
 contains a bug in its sample code — it uses the wrong repo_id.
 This script uses the CORRECT repo_id.

 VisDrone classes (different from COCO — do not mix):
   0: pedestrian    5: truck
   1: people        6: tricycle
   2: bicycle       7: awning-tricycle
   3: car           8: bus
   4: van           9: motor

 Usage:
   python3 model_download.py
   python3 model_download.py --also-download-small
=============================================================
"""

import sys
import os
import argparse
import time

def download_model(repo_id: str,
                   filename: str,
                   output_name: str) -> str:
    """Download a model from HuggingFace and return local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("[ERROR] huggingface_hub not installed.")
        print("  Run: pip install huggingface-hub")
        sys.exit(1)

    print(f"[DOWNLOAD] Repo     : {repo_id}")
    print(f"[DOWNLOAD] File     : {filename}")
    print(f"[DOWNLOAD] Save as  : {output_name}")
    print()

    t_start = time.time()
    try:
        path = hf_hub_download(
            repo_id  = repo_id,
            filename = filename,
        )
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print()
        print("Possible causes:")
        print("  1. No internet connection")
        print("  2. Model repo is private or has been removed")
        print("  3. huggingface_hub cache permission issue")
        print("     Try: export HF_HOME=/tmp/hf_cache")
        sys.exit(1)

    elapsed = time.time() - t_start

    # Copy to working directory with a clean name
    import shutil
    shutil.copy(path, output_name)
    size_mb = os.path.getsize(output_name) / 1024 / 1024
    print(f"[DOWNLOAD] Complete in {elapsed:.1f}s — {size_mb:.1f}MB → {output_name}")
    return output_name


def verify_model(model_path: str):
    """Load the model and verify it detects the correct VisDrone classes."""
    print(f"\n[VERIFY] Loading {model_path}...")
    try:
        from ultralytics import YOLO
        import numpy as np
    except ImportError:
        print("[ERROR] ultralytics not installed: pip install ultralytics")
        sys.exit(1)

    model = YOLO(model_path)

    # Expected VisDrone class names
    expected = {
        0: "pedestrian", 1: "people",    2: "bicycle",
        3: "car",        4: "van",       5: "truck",
        6: "tricycle",   7: "awning-tricycle",
        8: "bus",        9: "motor",
    }

    print("\n[VERIFY] Class mapping:")
    all_ok = True
    for cid, name in model.names.items():
        expected_name = expected.get(cid, "UNEXPECTED")
        ok = name == expected_name
        status = "✓" if ok else "✗"
        print(f"  {status}  [{cid}] {name:25s} "
              f"{'← OK' if ok else f'← EXPECTED: {expected_name}'}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\n[WARN] Class mapping mismatch detected.")
        print("  This may mean the model was updated or is a different variant.")
        print("  Update TARGET_CLASSES in config.yaml to match the above.")
    else:
        print("\n[VERIFY] ✓ All 10 VisDrone classes correctly mapped.")

    # Quick inference test
    print("\n[VERIFY] Running inference speed test (100 frames)...")
    import time
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    # Warm up
    for _ in range(3):
        model(dummy, verbose=False)
    t = time.time()
    for _ in range(100):
        model(dummy, verbose=False)
    elapsed = time.time() - t
    fps = 100 / elapsed
    latency = elapsed / 100 * 1000

    print(f"  Throughput : {fps:.1f} FPS")
    print(f"  Latency    : {latency:.1f}ms")

    if fps < 5:
        print("  [WARN] Very low FPS — CPU-only inference detected.")
        print("         Expected: 8-15 FPS on CPU, 30-120 FPS with GPU/NPU.")
    elif fps < 15:
        print("  [OK] CPU inference — acceptable for SITL testing.")
    else:
        print("  [OK] Hardware-accelerated inference detected.")

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Download VisDrone-trained YOLO11 models")
    parser.add_argument('--also-download-small',
                        action='store_true',
                        help='Also download yolo11s VisDrone variant '
                             '(better person detection, slightly slower)')
    parser.add_argument('--skip-verify',
                        action='store_true',
                        help='Skip inference speed test after download')
    args = parser.parse_args()

    print("="*60)
    print(" Drone Tracker — VisDrone Model Download")
    print("="*60)
    print()
    print(" IMPORTANT: VisDrone uses different class IDs from COCO.")
    print(" config.yaml already has the correct VisDrone mappings.")
    print(" Do NOT mix this model with COCO class IDs.")
    print()

    # ── YOLO11n VisDrone (primary — fast, deployment-ready) ──
    primary = download_model(
        repo_id     = "dronefreak/visdrone-yolov11n",  # CORRECT ID
        filename    = "best.pt",
        output_name = "visdrone_yolo11n.pt",
    )

    # ── YOLO11s VisDrone (optional — better person detection) ──
    if args.also_download_small:
        print()
        print("[INFO] Downloading YOLO11s variant "
              "(32.3% mAP@50 vs 27.59% for nano)...")
        print("[INFO] Better pedestrian detection at ~2x inference cost.")
        # Check if s variant exists in same zoo
        try:
            download_model(
                repo_id     = "dronefreak/visdrone-yolov11s",
                filename    = "best.pt",
                output_name = "visdrone_yolo11s.pt",
            )
        except SystemExit:
            print("[INFO] yolo11s not found under that repo.")
            print("  Check https://huggingface.co/collections/dronefreak/"
                  "visdrone-detection-model-zoo for the correct s-model repo.")

    if not args.skip_verify:
        verify_model(primary)

    print()
    print("="*60)
    print(" Download complete.")
    print()
    print(" Update config.yaml:")
    print("   model:")
    print(f"     path: \"{primary}\"")
    print()
    print(" The VisDrone class groups in config.yaml are already")
    print(" configured correctly for this model.")
    print("="*60)


if __name__ == '__main__':
    main()
