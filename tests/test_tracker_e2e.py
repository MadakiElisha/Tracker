"""
End-to-end run of the real tracker.py main loop in a subprocess, with
a stubbed `ultralytics` (fixed detection), a synthetic video pattern, no FC
(--no-fly) and no display. Proves the wiring: config -> state -> PID -> GCS tap
-> run log -> analyzer -> clean shutdown. It does NOT prove detection quality
or flight behaviour.
"""
import csv, json, os, signal, socket, subprocess, sys, textwrap, time, urllib.request
import pytest
import yaml

pytest.importorskip("cv2")
pytest.importorskip("flask")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import analyze_run as ar

FAKE_ULTRALYTICS = textwrap.dedent('''
    import numpy as np
    class _T:
        def __init__(self, a): self.a = a
        def cpu(self): return self
        def numpy(self): return self.a
    class _Boxes:
        def __init__(self, vis):
            self.id   = _T(np.array([1.0])) if vis else None
            self.xyxy = _T(np.array([[270.0, 165.0, 370.0, 315.0]]))
            self.conf = _T(np.array([0.9]))
            self.cls  = _T(np.array([0.0]))
    class _R:
        def __init__(self, vis): self.boxes = _Boxes(vis)
    class YOLO:
        def __init__(self, path): pass
        def track(self, frame, **kw): return [_R(True)]
''')


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


@pytest.fixture
def env(tmp_path):
    stub = tmp_path / "stub" / "ultralytics"
    stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(FAKE_ULTRALYTICS)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    cfg['gcs']['port'] = free_port()
    cfg['model']['path'] = str(tmp_path / "best.pt")
    cfg['run_log']['dir'] = str(tmp_path / "runs")
    cfg['logging']['file'] = str(tmp_path / "t.log")
    cfg['video']['source'] = str(tmp_path / "missing.mp4")     # -> synthetic pattern
    cfgp = tmp_path / "cfg.yaml"
    cfgp.write_text(yaml.safe_dump(cfg))
    (tmp_path / "best.pt").write_bytes(b"x")
    e = dict(os.environ, PYTHONPATH=f"{tmp_path/'stub'}{os.pathsep}{ROOT}", PYTHONUNBUFFERED="1")
    return tmp_path, cfg, cfgp, e


def start(cfgp, env, *extra):
    return subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tracker.py"), "--config", str(cfgp),
         "--connect", "127.0.0.1:1", "--no-fly", "--no-display", *extra],
        env=env, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def post_tap(port, x, y, tries=40):
    for _ in range(tries):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/tap",
                                         data=json.dumps({"x": x, "y": y}).encode(),
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req, timeout=1).read()
        except Exception:
            time.sleep(0.25)
    raise AssertionError("GCS server never came up")


def test_full_loop_tap_lock_and_clean_shutdown(env):
    tmp, cfg, cfgp, e = env
    p = start(cfgp, e)
    try:
        post_tap(cfg['gcs']['port'], 320, 240)       # inside the fixed box
        time.sleep(2.5)
    finally:
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=20)
    assert p.returncode == 0, out[-2000:]
    assert "Traceback" not in out, out[-2000:]
    assert "Locked" in out and "GCS_TAP" in out

    runs = list((tmp / "runs").glob("run_*.csv"))
    assert len(runs) == 1
    rows = list(csv.DictReader(open(runs[0])))
    assert ar.check_log_integrity(rows)[0] == ar.PASS
    locked = [r for r in rows if r["locked_id"] == "1" and r["target_visible"] == "1"]
    assert len(locked) > 10, "lock never took effect"
    # box area ~0.049 vs pedestrian setpoint 0.01 -> too close -> back off, centred -> no yaw
    assert float(locked[-1]["cmd_vx"]) < 0
    assert abs(float(locked[-1]["cmd_yaw"])) < 0.05
    assert rows[-1]["errors"] in ("0", "")


def test_tap_outside_any_target_does_not_lock(env):
    tmp, cfg, cfgp, e = env
    p = start(cfgp, e)
    try:
        post_tap(cfg['gcs']['port'], 20, 20)
        time.sleep(1.5)
    finally:
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=20)
    assert p.returncode == 0
    rows = list(csv.DictReader(open(next((tmp / "runs").glob("run_*.csv")))))
    assert not any(r["locked_id"] for r in rows)


def test_missing_model_exits_nonzero_with_hint(env):
    tmp, cfg, cfgp, e = env
    (tmp / "best.pt").unlink()
    p = start(cfgp, e)
    out, _ = p.communicate(timeout=30)
    assert p.returncode == 1
    assert "model_download.py" in out


def test_refuses_to_start_without_fc_unless_no_fly(env):
    tmp, cfg, cfgp, e = env
    p = subprocess.Popen(
        [sys.executable, os.path.join(ROOT, "tracker.py"), "--config", str(cfgp),
         "--connect", "127.0.0.1:1", "--no-display"],
        env=e, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out, _ = p.communicate(timeout=60)
    assert p.returncode == 1
    assert "refusing to start" in out


FAKE_SWAP = textwrap.dedent('''
    import os, time
    import numpy as np
    FLAG = os.environ["SWAP_FLAG"]
    _t0 = [None]
    class _T:
        def __init__(self, a): self.a = a
        def cpu(self): return self
        def numpy(self): return self.a
    class _Boxes:
        def __init__(self, tid):
            self.id   = _T(np.array([float(tid)])) if tid else None
            self.xyxy = _T(np.array([[270.0, 165.0, 370.0, 315.0]]))
            self.conf = _T(np.array([0.9]))
            self.cls  = _T(np.array([0.0]))
    class _R:
        def __init__(self, tid): self.boxes = _Boxes(tid)
    class YOLO:
        def __init__(self, path): pass
        def track(self, frame, **kw):
            if os.path.exists(FLAG):
                _t0[0] = _t0[0] or time.monotonic()
                age = time.monotonic() - _t0[0]
                if age < 0.5:
                    return [_R(None)]            # detector drops the target for 0.5 s ...
                return [_R(7)]                   # ... and ByteTrack comes back with a NEW id
            return [_R(1)]
''')


def test_reid_relocks_after_track_id_change(env):
    tmp, cfg, cfgp, e = env
    (tmp / "stub" / "ultralytics" / "__init__.py").write_text(FAKE_SWAP)
    flag = tmp / "swap.flag"
    e = dict(e, SWAP_FLAG=str(flag))
    # the synthetic video's patch moves relative to the fixed stub box, so colour match is not meaningful
    # here; appearance scoring is covered by tests/test_reid.py. This test proves the wiring.
    cfg['tracking']['reid'] = {'min_appearance': 0.0, 'threshold': 0.4}
    cfgp.write_text(yaml.safe_dump(cfg))
    p = start(cfgp, e)
    try:
        post_tap(cfg['gcs']['port'], 320, 240)
        time.sleep(1.5)
        flag.write_text("go")
        time.sleep(3.0)
    finally:
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=20)
    assert p.returncode == 0 and "Traceback" not in out, out[-2000:]
    assert "[REID] re-locked" in out and "id 1->7" in out, out[-2000:]
    rows = list(csv.DictReader(open(next((tmp / "runs").glob("run_*.csv")))))
    assert any(r["reid"] == "search" for r in rows) and any(r["reid"] == "relocked" for r in rows)
    assert rows[-1]["locked_id"] == "7" and rows[-1]["target_visible"] == "1"


def test_reid_can_be_disabled(env):
    tmp, cfg, cfgp, e = env
    cfg['tracking']['reid'] = {'enabled': False}
    cfg['tracking']['max_lost_frames'] = 5
    cfgp.write_text(yaml.safe_dump(cfg))
    (tmp / "stub" / "ultralytics" / "__init__.py").write_text(FAKE_SWAP)
    flag = tmp / "swap.flag"
    e = dict(e, SWAP_FLAG=str(flag))
    p = start(cfgp, e)
    try:
        post_tap(cfg['gcs']['port'], 320, 240)
        time.sleep(1.5)
        flag.write_text("go")
        time.sleep(3.0)
    finally:
        p.send_signal(signal.SIGINT)
        out, _ = p.communicate(timeout=20)
    assert "[REID]" not in out
    rows = list(csv.DictReader(open(next((tmp / "runs").glob("run_*.csv")))))
    assert rows[-1]["locked_id"] == ""            # old behaviour: lock dropped, nothing re-acquired
