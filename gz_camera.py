"""Gazebo camera source with a cv2.VideoCapture-like API.

Use as  --source gz:/drone/camera . Needs python3-gz-transport13 and
python3-gz-msgs10 (Gazebo Harmonic). The conversion is pure and unit-tested;
the transport wiring is UNTESTED against a live Gazebo.
"""
import threading
import time

import numpy as np

from tracker_logger import get_logger

log = get_logger(__name__)

# gz.msgs PixelFormatType enum values
_RGB_INT8, _BGR_INT8, _L_INT8, _RGBA_INT8 = 3, 8, 1, 4  # fallbacks; see below


def image_msg_to_bgr(msg):
    """Convert a gz Image-like message (width,height,data,pixel_format_type) to BGR ndarray."""
    w, h = int(msg.width), int(msg.height)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    n = w * h
    if n == 0:
        raise ValueError("empty image")
    ch = buf.size // n
    if buf.size != n * ch or ch not in (1, 3, 4):
        raise ValueError(f"unexpected image buffer size {buf.size} for {w}x{h}")
    img = buf.reshape(h, w, ch)
    fmt = getattr(msg, "pixel_format_type", None)
    if ch == 1:
        return np.repeat(img, 3, axis=2)
    if ch == 4:
        img = img[:, :, :3]
    if fmt == _BGR_INT8:
        return img.copy()
    return img[:, :, ::-1].copy()  # RGB -> BGR


class GzCamera:
    def __init__(self, topic="/drone/camera"):
        from gz.msgs10.image_pb2 import Image
        from gz.transport13 import Node
        self._node = Node()
        self._lock = threading.Lock()
        self._frame = None
        self._seq = 0
        self._last_read = 0
        self._ok = self._node.subscribe(Image, topic, self._cb)
        if not self._ok:
            log.error("could not subscribe to %s", topic)

    def _cb(self, msg):
        try:
            f = image_msg_to_bgr(msg)
        except Exception as e:  # keep the transport thread alive
            log.warning("bad gz image: %s", e)
            return
        with self._lock:
            self._frame, self._seq = f, self._seq + 1

    def isOpened(self):
        return self._ok

    def read(self, timeout=1.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._seq != self._last_read and self._frame is not None:
                    self._last_read = self._seq
                    return True, self._frame.copy()
            time.sleep(0.005)
        return False, None

    def set(self, *_a):
        return False

    def get(self, _prop):
        return 0.0

    def release(self):
        self._ok = False
