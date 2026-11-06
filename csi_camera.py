#!/usr/bin/env python3
"""
Raspberry Pi ribbon (CSI) camera source for tracker.py:  --source csi:0

On a Pi 5, OpenCV cannot open CSI cameras (they go through libcamera), so this
wraps Picamera2 behind the small cv2.VideoCapture-like interface tracker.py uses:
isOpened(), read() -> (ok, BGR frame), set(), release().

Works with any libcamera sensor (IMX296, IMX519, IMX477, ...). For tracking from
a moving drone it:
  * locks focus (autofocus hunting mid-flight blurs the target): lens at
    infinity unless camera.csi.lens_position is given (dioptres, 0 = infinity)
  * optionally fixes a short shutter (camera.csi.max_exposure_us); gain stays automatic
  * delivers frames already in BGR order at the size tracker.py asks for

Picamera2 comes from apt on Raspberry Pi OS (python3-picamera2), so run the
tracker from a venv made with --system-site-packages (scripts/setup_pi.sh).
"""

from __future__ import annotations

import threading
import time

try:                                   # only exists on a Raspberry Pi
    from picamera2 import Picamera2    # type: ignore
except Exception:                      # pragma: no cover - exercised on the Pi
    Picamera2 = None


def parse_csi_spec(spec: str) -> int:
    """'csi:' / 'csi' / 'csi:0' / 'csi:1' -> camera index."""
    rest = spec.split(':', 1)[1] if ':' in spec else ''
    rest = rest.strip()
    return int(rest) if rest else 0


class CsiCamera:
    """Picamera2 capture with a background thread that always holds the newest frame.

    The tracker must never process an old frame (it is a control loop), so a
    single-slot buffer is used instead of a queue: read() returns the latest
    frame, or waits briefly for the next one.
    """

    is_live = True        # tracker.py: do not try to rewind a live camera

    def __init__(self, index: int = 0, width: int = 640, height: int = 480,
                 fps: float = 30.0, cfg: dict | None = None, picamera_cls=None):
        cfg = dict(cfg or {})
        cls = picamera_cls or Picamera2
        if cls is None:
            raise ImportError("Picamera2 not available (sudo apt install python3-picamera2, "
                              "and use a venv created with --system-site-packages)")
        self.width, self.height = int(width), int(height)
        self._cam = cls(index)
        controls = {"FrameRate": float(cfg.get('fps', fps))}
        self.max_exposure_us = int(cfg.get('max_exposure_us', 0) or 0)
        config = self._cam.create_video_configuration(
            main={"size": (self.width, self.height), "format": "RGB888"},   # RGB888 is BGR in memory
            buffer_count=4, controls=controls)
        self._cam.configure(config)
        self._apply_focus(cfg)
        self._cam.start()
        if self.max_exposure_us:
            self._limit_exposure()
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._frame = None
        self._frame_t = 0.0
        self._running = True
        self._err = None
        self._thread = threading.Thread(target=self._loop, name="csi_camera", daemon=True)
        self._thread.start()

    # -- camera controls ----------------------------------------------------
    def _apply_focus(self, cfg: dict) -> None:
        props = getattr(self._cam, 'camera_controls', {}) or {}
        if 'AfMode' not in props:
            return                                   # fixed-focus sensor (IMX296, IMX477)
        lens = float(cfg.get('lens_position', 0.0))  # dioptres; 0.0 = infinity
        self._cam.set_controls({"AfMode": 0, "LensPosition": lens})   # 0 = manual

    def _limit_exposure(self) -> None:
        # Fixed short shutter, gain left automatic: outdoors this keeps motion blur to a
        # few milliseconds while brightness still adapts. (On libcamera older than 0.4,
        # setting ExposureTime also freezes gain; check the image on the bench.)
        try:
            self._cam.set_controls({"ExposureTime": self.max_exposure_us})
        except Exception:
            pass

    # -- capture thread -----------------------------------------------------
    def _loop(self) -> None:
        while self._running:
            try:
                arr = self._cam.capture_array("main")
            except Exception as e:                   # camera unplugged, driver error
                # No new frame: read() times out and the tracker treats it as video loss.
                # Never hand the last good frame out again as if it were fresh.
                self._err = e
                with self._lock:
                    self._frame = None
                time.sleep(0.05)
                continue
            if arr is not None and arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, :3]
            with self._lock:
                self._frame = arr
                self._frame_t = time.monotonic()
            self._new.set()

    # -- VideoCapture-like API ------------------------------------------------
    def isOpened(self) -> bool:
        return self._running

    def read(self, timeout: float = 0.5):
        if not self._new.wait(timeout):
            return False, None
        self._new.clear()
        with self._lock:
            frame = self._frame
        if frame is None:
            return False, None
        return True, frame

    def frame_age(self) -> float:
        with self._lock:
            return time.monotonic() - self._frame_t if self._frame_t else float('inf')

    def set(self, *_a) -> bool:            # cv2 compatibility; a live camera cannot seek
        return False

    def release(self) -> None:
        self._running = False
        try:
            self._cam.stop()
            self._cam.close()
        except Exception:
            pass
