#!/usr/bin/env python3
"""
Hardware-in-the-loop frame relay (runs on the PC next to Gazebo).

Serves the drone camera to the Pi over TCP so the real tracker runs on the Pi + Hailo against
the simulated world:

    python3 sim/frame_relay.py                                  # gz:/drone/camera -> 0.0.0.0:5601
    python3 sim/frame_relay.py --source clip.mp4 --loop         # test without Gazebo
    # on the Pi:  python3 tracker.py --source net:10.42.0.1:5601 --connect 0.0.0.0:14550 ...

Only the NEWEST frame is ever sent: if the link or the Pi is slow, old frames are dropped rather
than queued (queued frames would add latency to a control loop). One client at a time; a new
connection replaces the old one. Prints frames/s, JPEG size and drops every 5 s.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from net_camera import encode_frame   # noqa: E402


class Latest:
    """Single-slot mailbox: put() replaces, get() waits for something newer than last time."""

    def __init__(self):
        self._cv = threading.Condition()
        self._item, self._seq = None, 0

    def put(self, item):
        with self._cv:
            self._item, self._seq = item, self._seq + 1
            self._cv.notify_all()

    def get(self, last_seq, timeout=1.0):
        with self._cv:
            self._cv.wait_for(lambda: self._seq != last_seq, timeout)
            return self._seq, self._item


def open_source(spec: str):
    if spec.startswith('gz:'):
        from gz_camera import GzCamera
        return GzCamera(spec[3:] or '/drone/camera')
    import cv2
    return cv2.VideoCapture(int(spec) if spec.isdigit() else spec)


def capture_loop(src, box: Latest, quality: int, size, loop: bool, stats: dict, stop: threading.Event,
                 pace_fps: float):
    import cv2
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    period = 1.0 / pace_fps if pace_fps > 0 else 0.0
    while not stop.is_set():
        t0 = time.monotonic()
        ok, frame = src.read()
        if not ok:
            if loop and hasattr(src, 'set'):
                src.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.01)
            continue
        t_cap = time.time()
        if size and (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        ok, jpg = cv2.imencode('.jpg', frame, params)
        if ok:
            box.put((t_cap, jpg.tobytes()))
            stats['captured'] += 1
        if period:                                   # files play as fast as they decode otherwise
            time.sleep(max(0.0, period - (time.monotonic() - t0)))


def serve(box: Latest, host: str, port: int, stats: dict, stop: threading.Event):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    srv.settimeout(0.5)
    print(f"[relay] listening on {host}:{port}", flush=True)
    while not stop.is_set():
        try:
            conn, addr = srv.accept()
        except socket.timeout:
            continue
        print(f"[relay] client {addr[0]} connected", flush=True)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        last = 0
        try:
            while not stop.is_set():
                seq, item = box.get(last)
                if item is None or seq == last:
                    continue
                stats['dropped'] += max(0, seq - last - 1) if last else 0
                last = seq
                t_cap, jpg = item
                conn.sendall(encode_frame(jpg, t_cap))
                stats['sent'] += 1
                stats['bytes'] += len(jpg)
        except (BrokenPipeError, ConnectionResetError, OSError):
            print(f"[relay] client {addr[0]} disconnected", flush=True)
        finally:
            conn.close()
    srv.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='gz:/drone/camera', help='gz:/topic, a video file or a camera index')
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=5601)
    ap.add_argument('--quality', type=int, default=90, help='JPEG quality (90: detection unaffected, ~40 KB)')
    ap.add_argument('--size', default='640x480', help='WxH sent to the Pi (the tracker runs at 640x480)')
    ap.add_argument('--loop', action='store_true', help='loop a video file')
    ap.add_argument('--fps', type=float, default=30.0, help='pace file playback (Gazebo paces itself)')
    ap.add_argument('--seconds', type=float, default=0.0, help='stop after this long (tests); 0 = run until Ctrl+C')
    a = ap.parse_args()

    size = tuple(int(x) for x in a.size.lower().split('x')) if a.size else None
    src = open_source(a.source)
    if not src.isOpened():
        print(f"[relay] cannot open {a.source}")
        return 1
    box, stop = Latest(), threading.Event()
    stats = dict(captured=0, sent=0, dropped=0, bytes=0)
    pace = 0.0 if a.source.startswith('gz:') else a.fps
    threading.Thread(target=capture_loop, args=(src, box, a.quality, size, a.loop, stats, stop, pace),
                     daemon=True).start()
    threading.Thread(target=serve, args=(box, a.host, a.port, stats, stop), daemon=True).start()
    t0, last = time.monotonic(), dict(stats)
    try:
        while not stop.is_set():
            time.sleep(0.25)
            el = time.monotonic() - t0
            if a.seconds and el >= a.seconds:
                break
            if el >= 5.0:
                d = {k: stats[k] - last[k] for k in stats}
                kb = d['bytes'] / max(d['sent'], 1) / 1024
                print(f"[relay] {d['captured'] / el:.1f} fps captured, {d['sent'] / el:.1f} fps sent, "
                      f"{kb:.0f} KB/frame, {d['dropped']} dropped", flush=True)
                t0, last = time.monotonic(), dict(stats)
    except KeyboardInterrupt:
        pass
    stop.set()
    return 0


if __name__ == '__main__':
    sys.exit(main())
