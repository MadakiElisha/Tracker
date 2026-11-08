"""HITL frame relay (PC) -> NetCamera (Pi), over real localhost TCP."""
import os
import socket
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from net_camera import NetCamera, encode_frame, read_message, parse_net_spec   # noqa: E402


def _free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def _jpeg(val, w=64, h=48):
    img = np.full((h, w, 3), val, np.uint8)
    return cv2.imencode('.jpg', img)[1].tobytes()


def test_parse_spec():
    assert parse_net_spec('net:10.42.0.1:5601') == ('10.42.0.1', 5601)
    assert parse_net_spec('net:pc.local') == ('pc.local', 5601)


def test_message_roundtrip_and_bad_magic():
    a, b = socket.socketpair()
    a.sendall(encode_frame(b'\xff\xd8abc', 123.5))
    t, data = read_message(b)
    assert t == 123.5 and data == b'\xff\xd8abc'
    a.sendall(b'XXXX' + b'\0' * 12)
    with pytest.raises(ValueError, match="magic"):
        read_message(b)
    a.close(); b.close()


class FakeRelay:
    """Minimal server: sends frames of increasing brightness until told to drop the client."""

    def __init__(self, port):
        self.srv = socket.socket()
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(('127.0.0.1', port))
        self.srv.listen(1)
        self.stop = threading.Event()
        self.drop = threading.Event()
        self.clients = 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.srv.settimeout(0.2)
        while not self.stop.is_set():
            try:
                c, _ = self.srv.accept()
            except socket.timeout:
                continue
            self.clients += 1
            self.drop.clear()
            v = 10
            try:
                while not self.stop.is_set() and not self.drop.is_set():
                    c.sendall(encode_frame(_jpeg(v)))
                    v = min(250, v + 10)
                    time.sleep(0.02)
            except OSError:
                pass
            c.close()

    def close(self):
        self.stop.set()
        self.srv.close()


def test_netcamera_receives_newest_frames_and_reconnects():
    port = _free_port()
    relay = FakeRelay(port)
    cam = NetCamera('127.0.0.1', port)
    try:
        ok, f = cam.read(timeout=3.0)
        assert ok and f.shape == (48, 64, 3) and cam.connected
        time.sleep(0.3)                              # ~15 frames arrive meanwhile
        ok, f = cam.read(timeout=1.0)
        assert ok and f.mean() > 60                  # a recent frame, not the first one
        assert cam.latency_s is not None and abs(cam.latency_s) < 1.0
        relay.drop.set()                             # relay drops the client...
        deadline = time.time() + 5
        while relay.clients < 2 and time.time() < deadline:
            time.sleep(0.05)
        assert relay.clients >= 2                    # ...and NetCamera reconnected
        ok, _ = cam.read(timeout=3.0)
        assert ok
    finally:
        cam.release()
        relay.close()


def test_netcamera_without_relay_gives_no_frames():
    cam = NetCamera('127.0.0.1', _free_port(), connect_timeout=0.2)
    try:
        ok, f = cam.read(timeout=0.5)
        assert not ok and f is None and not cam.connected
    finally:
        cam.release()


def test_frame_relay_script_end_to_end(tmp_path):
    """sim/frame_relay.py serving a video file -> NetCamera, as on the PC/Pi pair."""
    clip = tmp_path / 'clip.avi'
    w = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'MJPG'), 30, (640, 480))
    for i in range(30):
        w.write(np.full((480, 640, 3), i * 8, np.uint8))
    w.release()
    port = _free_port()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, 'sim', 'frame_relay.py'), '--source', str(clip),
                             '--loop', '--host', '127.0.0.1', '--port', str(port), '--seconds', '8'],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 5
        cam = None
        while time.time() < deadline:
            try:
                socket.create_connection(('127.0.0.1', port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        cam = NetCamera('127.0.0.1', port)
        got = 0
        t_end = time.time() + 2.0
        while time.time() < t_end:
            ok, f = cam.read(timeout=1.0)
            if ok:
                assert f.shape == (480, 640, 3)
                got += 1
        assert got >= 20, got                         # ~30 fps paced playback
    finally:
        if cam:
            cam.release()
        proc.terminate()
        proc.wait(5)
