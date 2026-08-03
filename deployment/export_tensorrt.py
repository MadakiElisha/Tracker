#!/usr/bin/env python3
"""
=============================================================
 Export YOLO11n to TensorRT engine for Jetson Orin Nano/NX

 Run this ON the Jetson — TensorRT engines are hardware-specific.
 A .engine built on Orin Nano will not run on a different GPU.

 Expected output: yolo11n.engine (~15-30MB)
 Expected time:   5-15 minutes on first run
 Expected perf:   80-120 FPS on Orin Nano (FP16, 640px)
=============================================================
"""

import sys
import os
import time

# Verify we're on a Jetson
if not os.path.exists('/usr/bin/nvpmodel'):
    print("[ERROR] This script must run on a Jetson device.")
    print("  Detected platform is not Jetson (no nvpmodel found).")
    sys.exit(1)

# Verify CUDA is available
try:
    import torch
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available. Check JetPack installation.")
        sys.exit(1)
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("[ERROR] PyTorch not found. Install JetPack with Deep Learning components.")
    sys.exit(1)

from ultralytics import YOLO

print("="*55)
print(" YOLO11n → TensorRT FP16 Export")
print("="*55)
print(f" GPU: {torch.cuda.get_device_name(0)}")
print(f" Input resolution: 640x640")
print(f" Precision: FP16")
print(f" Workspace: 4GB")
print()
print(" This will take 5-15 minutes. Do not interrupt.")
print("="*55)

model = YOLO("yolo11n.pt")

t_start = time.time()
model.export(
    format    = "engine",
    device    = 0,
    half      = True,        # FP16 — near-identical accuracy, ~2x faster
    imgsz     = 640,
    workspace = 4,           # GB of GPU RAM for TensorRT optimization
    verbose   = True,
    batch     = 1,
)
elapsed = time.time() - t_start
print(f"\n[EXPORT] Complete in {elapsed:.0f}s")

# Verify and benchmark
engine_path = "yolo11n.engine"
if not os.path.exists(engine_path):
    print(f"[ERROR] Expected engine file not found: {engine_path}")
    sys.exit(1)

size_mb = os.path.getsize(engine_path) / 1024 / 1024
print(f"[EXPORT] Engine size: {size_mb:.1f}MB")

print("\n[BENCHMARK] Loading engine and benchmarking...")
import numpy as np
engine_model = YOLO(engine_path, task="detect")

dummy = np.zeros((480, 640, 3), dtype=np.uint8)

# Warm up
print("  Warming up (5 passes)...")
for _ in range(5):
    engine_model(dummy, verbose=False)

# Benchmark
N = 200
print(f"  Running {N} inference passes...")
t = time.time()
for _ in range(N):
    engine_model(dummy, verbose=False)
total = time.time() - t

fps     = N / total
latency = total / N * 1000

print(f"\n[RESULT] Throughput : {fps:.1f} FPS")
print(f"[RESULT] Latency    : {latency:.1f}ms per frame")
print(f"\n[DONE] yolo11n.engine is ready for deployment.")
