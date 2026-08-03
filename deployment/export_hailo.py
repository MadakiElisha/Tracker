#!/usr/bin/env python3
"""
=============================================================
 Export YOLO11n to Hailo HEF for RPi 5 + Hailo-8L AI Hat

 Can run on:
   A) Dev PC with Hailo Dataflow Compiler installed (recommended)
   B) RPi 5 directly (slower, 30-90 min)

 Output: yolo11n.hef (~5-15MB)
 Deploy: scp yolo11n.hef drone@dronetracker.local:~/drone_tracker/

 Expected perf on Hailo-8L: 30-45 FPS, 22-33ms latency
=============================================================
"""

import sys
import os
import time

print("="*55)
print(" YOLO11n → Hailo HEF Export")
print("="*55)

# Check Ultralytics is installed
try:
    from ultralytics import YOLO
    import ultralytics
    print(f"[OK] Ultralytics {ultralytics.__version__}")
except ImportError:
    print("[ERROR] Ultralytics not found: pip install ultralytics")
    sys.exit(1)

# Check Hailo Dataflow Compiler
hailo_dfc_available = False
try:
    import hailo_sdk_client
    hailo_dfc_available = True
    print(f"[OK] Hailo Dataflow Compiler available")
except ImportError:
    print("[WARN] Hailo Dataflow Compiler not found.")
    print("  Download from: https://developer.hailo.ai/developer-zone/")
    print("  Install: pip install hailo_dataflow_compiler-*.whl")
    print()
    print("  Alternatively, Ultralytics may handle the export internally.")
    print("  Attempting export via Ultralytics...")

print(f"\n Input resolution: 640x640")
print(f" Quantization: INT8 (Hailo native)")
print(f" Batch size: 1")
print()
print(" This will take 10-30 minutes. Do not interrupt.")
print("="*55)

model = YOLO("yolo11n.pt")

t_start = time.time()
try:
    result = model.export(
        format  = "hailo",
        imgsz   = 640,
        half    = True,   # INT8 quantization for Hailo
        batch   = 1,
    )
    elapsed = time.time() - t_start
    print(f"\n[EXPORT] Complete in {elapsed:.0f}s")

except Exception as e:
    print(f"\n[ERROR] Export failed: {e}")
    print()
    print("Possible causes:")
    print("  1. Hailo Dataflow Compiler not installed")
    print("     Download from https://developer.hailo.ai/developer-zone/")
    print("  2. Ultralytics version too old (need 8.3+)")
    print("     pip install --upgrade ultralytics")
    print("  3. Insufficient RAM for compilation (need 8GB+)")
    sys.exit(1)

# Find the output file
hef_candidates = [
    "yolo11n.hef",
    "yolo11n_hailo.hef",
    "runs/detect/yolo11n.hef",
]
hef_path = None
for candidate in hef_candidates:
    if os.path.exists(candidate):
        hef_path = candidate
        break

if hef_path is None:
    print("[WARN] Could not find output .hef file automatically.")
    print("  Check current directory and runs/ folder.")
    print("  Look for any .hef file and copy to drone_tracker/")
else:
    size_mb = os.path.getsize(hef_path) / 1024 / 1024
    print(f"[OK] HEF file: {hef_path} ({size_mb:.1f}MB)")

    # Copy to working directory if not already there
    if hef_path != "yolo11n.hef":
        import shutil
        shutil.copy(hef_path, "yolo11n.hef")
        print(f"[OK] Copied to yolo11n.hef")

print()
print("[NEXT STEPS]")
print("  If exported on dev PC:")
print("    scp yolo11n.hef drone@dronetracker.local:~/drone_tracker/")
print()
print("  If already on RPi 5, verify with:")
print("    hailortcli parse-hef yolo11n.hef")
print()
print("  Then update tracker2bi.py:")
print("    YOLO_MODEL = 'yolo11n.hef'")
print("  Or use inference_backend.py for auto-detection.")
