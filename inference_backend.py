#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Inference Backend Selector

 Automatically detects hardware at runtime and returns the
 correct YOLO model path and camera interface.

 Platforms supported:
   Jetson Orin Nano/NX  → TensorRT .engine (CUDA FP16)
   RPi 5 + Hailo-8L     → HailoRT .hef     (NPU INT8)
   Dev PC / other       → PyTorch .pt       (CPU/CUDA)

 Usage in tracker2bi.py:
   from inference_backend import detect_platform, open_camera, read_frame
   model_path, platform = detect_platform()
   model = YOLO(model_path)
   cam, cam_type = open_camera(FRAME_W, FRAME_H)
=============================================================
"""

import os
import sys
import subprocess


def detect_platform() -> tuple:
    """
    Detect which hardware platform we're running on and return:
      (model_path: str, platform_name: str)

    Platform names: 'jetson' | 'rpi5_hailo' | 'dev'
    """

    # --- Jetson detection ---
    # nvpmodel is a Jetson-exclusive binary
    if os.path.exists('/usr/bin/nvpmodel'):
        engine = 'yolo11n.engine'
        if os.path.exists(engine):
            print(f"[BACKEND] Platform: Jetson Orin → TensorRT ({engine})")
            return engine, 'jetson'
        else:
            print(f"[BACKEND] Jetson detected, but {engine} not found.")
            print(f"  Export it with: python3 export_tensorrt.py")
            print(f"  Falling back to PyTorch (slow).")
            return 'yolo11n.pt', 'jetson_fallback'

    # --- Raspberry Pi 5 + Hailo detection ---
    # /dev/hailo0 only exists when the Hailo driver is loaded
    if os.path.exists('/dev/hailo0'):
        hef = 'yolo11n.hef'
        if os.path.exists(hef):
            print(f"[BACKEND] Platform: RPi 5 + Hailo-8L → HailoRT ({hef})")
            return hef, 'rpi5_hailo'
        else:
            print(f"[BACKEND] Hailo-8L detected, but {hef} not found.")
            print(f"  Export it with: python3 export_hailo.py")
            print(f"  Falling back to PyTorch (slow).")
            return 'yolo11n.pt', 'rpi5_fallback'

    # --- RPi without Hailo (CPU only) ---
    if os.path.exists('/proc/device-tree/model'):
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model_str = f.read()
            if 'Raspberry Pi 5' in model_str:
                print(f"[BACKEND] Platform: RPi 5 (CPU only — no Hailo detected)")
                print(f"  For better performance, install the Hailo-8L AI Hat.")
                return 'yolo11n.pt', 'rpi5_cpu'
        except Exception:
            pass

    # --- Development PC / SITL ---
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"[BACKEND] Platform: Dev PC with CUDA ({gpu}) → PyTorch")
        else:
            print(f"[BACKEND] Platform: Dev PC (CPU only) → PyTorch")
    except ImportError:
        print(f"[BACKEND] Platform: Unknown → PyTorch (CPU)")

    return 'yolo11n.pt', 'dev'


def get_fc_connection(platform: str) -> str:
    """
    Return the correct FC connection string for each platform.
    Override with --connect argument in tracker if needed.
    """
    connections = {
        'jetson':          '/dev/ttyTHS0',    # Jetson UART0
        'jetson_fallback': '/dev/ttyTHS0',
        'rpi5_hailo':      '/dev/serial0',    # RPi 5 UART
        'rpi5_fallback':   '/dev/serial0',
        'rpi5_cpu':        '/dev/serial0',
        'dev':             '127.0.0.1:14550', # SITL over UDP
    }
    return connections.get(platform, '127.0.0.1:14550')


# ============================================================
# Camera interface
# ============================================================

# Try importing Picamera2 (RPi only)
try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

import cv2
import numpy as np


def open_camera(frame_w: int = 640, frame_h: int = 480):
    """
    Open the camera appropriate for this platform.

    Returns: (camera_object, camera_type_string)
      camera_type: 'picamera2' | 'gstreamer' | 'opencv'
    """

    # --- Jetson: GStreamer + NvArgus pipeline ---
    if os.path.exists('/usr/bin/nvpmodel'):
        pipeline = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width=1280, height=720, "
            f"framerate=60/1, format=NV12 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={frame_w}, height={frame_h}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink drop=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[CAMERA] Jetson GStreamer pipeline open ({frame_w}x{frame_h}@60)")
            return cap, 'gstreamer'
        else:
            print("[CAMERA][WARN] GStreamer pipeline failed. "
                  "Check camera connection. Trying V4L2...")

    # --- RPi 5: Picamera2 ---
    if _PICAMERA2_AVAILABLE:
        try:
            cam = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (frame_w, frame_h), "format": "BGR888"},
                controls={"FrameRate": 60}
            )
            cam.configure(config)
            cam.start()
            print(f"[CAMERA] Picamera2 open ({frame_w}x{frame_h}@60)")
            return cam, 'picamera2'
        except Exception as e:
            print(f"[CAMERA][WARN] Picamera2 failed: {e}. Trying V4L2...")

    # --- Fallback: standard OpenCV V4L2 ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if cap.isOpened():
        print(f"[CAMERA] OpenCV V4L2 open ({frame_w}x{frame_h})")
        return cap, 'opencv'

    print("[CAMERA][ERROR] No camera could be opened.")
    return None, 'none'


def read_frame(cam, cam_type: str):
    """
    Read one frame from any camera backend.

    Returns: (success: bool, frame: np.ndarray or None)
    """
    if cam is None:
        return False, None

    if cam_type == 'picamera2':
        try:
            frame = cam.capture_array()
            return True, frame
        except Exception as e:
            print(f"[CAMERA] Read error: {e}")
            return False, None

    elif cam_type in ('opencv', 'gstreamer'):
        return cam.read()

    return False, None


def release_camera(cam, cam_type: str):
    """Release camera resources cleanly."""
    if cam is None:
        return
    if cam_type == 'picamera2':
        cam.stop()
    elif cam_type in ('opencv', 'gstreamer'):
        cam.release()


# ============================================================
# Quick self-test
# ============================================================
if __name__ == '__main__':
    print("="*55)
    print(" Inference Backend Detection")
    print("="*55)

    model_path, platform = detect_platform()
    fc_conn = get_fc_connection(platform)

    print(f"\n  Model path : {model_path}")
    print(f"  Platform   : {platform}")
    print(f"  FC connect : {fc_conn}")

    print("\n  Opening camera for 3s test...")
    cam, cam_type = open_camera(640, 480)

    if cam is not None:
        import time
        count = 0
        t_start = time.time()
        while time.time() - t_start < 3:
            ok, frame = read_frame(cam, cam_type)
            if ok:
                count += 1
        fps = count / 3
        print(f"  Camera FPS : {fps:.1f} ({cam_type})")
        release_camera(cam, cam_type)
    else:
        print("  Camera     : NOT AVAILABLE")

    print("\n" + "="*55)
