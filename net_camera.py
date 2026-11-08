#!/usr/bin/env python3
"""
Frames over the network for hardware-in-the-loop (HITL):  --source net:HOST:PORT

The PC runs Gazebo and sim/frame_relay.py; the Pi runs the real tracker on the Hailo with
--source net:10.42.0.1:5601, so detection, tracking, control and safety all run on the target
hardware against the simulated world. MAVLink comes separately (SITL --out to the Pi).

Wire format, one message per frame (all big-endian):
    b'TRK1' | float64 sender wall time (s) | uint32 JPEG length | JPEG bytes

NetCamera has the cv2.VideoCapture-like API tracker.py uses (isOpened, read, set, release),
keeps only the newest frame (a control loop must never process a backlog), reconnects if the
relay restarts, and reports how old frames are when they arrive (latency_s; meaningful when
both clocks are synced, e.g. chrony/NTP over the same link).
"""
from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np

MAGIC = b'TRK1'
HEADER = struct.Struct('>4sdI')            # magic, sender time, payload length
MAX_FRAME = 8 * 1024 * 1024


def encode_frame(jpeg: bytes, t: float | None = None) -> bytes:
    return HEADER.pack(MAGIC, time.time() if t is None else t, len(jpeg)) + jpeg


def _recv_exact(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("relay closed the connection")
        buf += chunk
    return bytes(buf)


def read_message(sock):
    """Read one frame message -> (sender_time, jpeg_bytes). Raises on a malformed stream."""
    magic, t, n = HEADER.unpack(_recv_exact(sock, HEADER.size))
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}: not a frame_relay stream")
    if n == 0 or n > MAX_FRAME:
        raise ValueError(f"bad frame length {n}")
    return t, _recv_exact(sock, n)


def parse_net_spec(spec: str):
    """'net:10.42.0.1:5601' -> ('10.42.0.1', 5601); 'net:host' -> (host, 5601)."""
    rest = spec.split(':', 1)[1] if spec.startswith('net:') else spec
    if ':' in rest:
        host, port = rest.rsplit(':', 1)
        return host, int(port)
    return rest, 5601


class NetCamera:
    is_live = True

    def __init__(self, host: str, port: int = 5601, connect_timeout: float = 3.0, decoder=None):
        import cv2
        self._decode = decoder or (lambda b: cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR))
        self.host, self.port = host, int(port)
        self.connect_timeout = connect_timeout
        self._lock = threading.Lock()
        self._new = threading.Event()
        self._frame = None
        self._frame_t = 0.0
        self.latency_s = None
        self.connected = False
        self.frames = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="net_camera", daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                with socket.create_connection((self.host, self.port), timeout=self.connect_timeout) as s:
                    s.settimeout(2.0)                  # no frame for 2 s -> reconnect
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.connected = True
                    while self._running:
                        t_sent, jpeg = read_message(s)
                        img = self._decode(jpeg)
                        if img is None:
                            continue
                        with self._lock:
                            self._frame = img
                            self._frame_t = time.monotonic()
                            self.latency_s = time.time() - t_sent
                            self.frames += 1
                        self._new.set()
            except Exception:
                self.connected = False
                with self._lock:
                    self._frame = None           # never hand out a frame from a dead link
                if self._running:
                    time.sleep(0.5)

    def isOpened(self) -> bool:
        return self._running

    def read(self, timeout: float = 0.5):
        if not self._new.wait(timeout):
            return False, None
        self._new.clear()
        with self._lock:
            f = self._frame
        return (f is not None), f

    def frame_age(self) -> float:
        with self._lock:
            return time.monotonic() - self._frame_t if self._frame_t else float('inf')

    def set(self, *_a) -> bool:
        return False

    def get(self, _prop) -> float:
        return 0.0

    def release(self) -> None:
        self._running = False
