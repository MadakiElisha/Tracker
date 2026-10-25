"""Exhibition launcher: timeline maths, command channel, report, and a real tracker driven by run_demo (no sim)."""
import csv, json, os, socket, subprocess, sys, textwrap
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ("", "sim", "demo"):
    sys.path.insert(0, os.path.join(ROOT, sub))
import show_paths as sp                     # noqa: E402
import demo_control as dc                   # noqa: E402
import report as rp                         # noqa: E402
import highway_paths as hp                  # noqa: E402

SHIPPED = os.path.join(ROOT, "demo", "scenarios", "exhibition.yaml")


def test_shipped_scenario_fits_road_and_runs_four_to_five_minutes():
    doc = yaml.safe_load(open(SHIPPED))
    segs = sp.load_segments(doc)
    assert segs[-1]["x0"] + segs[-1]["distance"] <= hp.ROAD_END_X
    assert 3.5 * 60 <= sp.show_duration(segs, lock_wait_s=8.0) <= 5.0 * 60
    assert all(b["x0"] > a["x0"] + a["distance"] for a, b in zip(segs, segs[1:]))   # never starts inside the previous


def test_scenario_too_long_for_road_is_refused():
    with pytest.raises(ValueError, match="road"):
        sp.load_segments({"segments": [{"actor": "car", "speed": 20, "run_s": 200}]})
    with pytest.raises(ValueError):
        sp.load_segments({"segments": [{"actor": "tank", "speed": 5, "run_s": 5}]})
    with pytest.raises(ValueError):
        sp.load_segments({"segments": []})


def test_actor_poses_only_one_actor_visible_and_standing_until_go():
    seg = sp.load_segments({"segments": [{"actor": "car", "speed": 10, "run_s": 20, "accel": 5}]})[0]
    standing = sp.actor_poses(seg, -1.0)
    assert standing["highway_car"][2] == 0.0 and standing["highway_car"][0] == seg["x0"]
    assert standing["highway_runner"][2] == sp.HIDDEN_Z
    moving = sp.actor_poses(seg, 10.0)
    assert moving["highway_car"][0] > seg["x0"] + 50


def test_vanish_and_lane_change_in_segments():
    seg = sp.load_segments({"segments": [{"actor": "person", "speed": 5, "run_s": 40, "vanish_after_s": 10,
                                          "vanish_for_s": 2, "lane_change_s": 8}]})[0]
    assert sp.actor_poses(seg, 5.0)["highway_runner"][2] == 0.0
    assert sp.actor_poses(seg, 11.0)["highway_runner"][2] == sp.HIDDEN_Z
    assert sp.actor_poses(seg, 13.0)["highway_runner"][2] == 0.0
    assert sp.actor_poses(seg, 20.0)["highway_runner"][1] != 8.5


def test_command_channel_ignores_old_commands_partial_lines_and_junk(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("lock person\n")                    # left over from an earlier show
    ch = dc.CommandChannel(str(p))
    assert ch.poll() == []
    with open(p, "a") as fh:
        fh.write("mark Segment one\nbogus x\nrel")
    assert ch.poll() == [("mark", "Segment one")]
    with open(p, "a") as fh:
        fh.write("ease\nlock car\n")
    assert ch.poll() == [("release", ""), ("lock", "car")]


def test_pick_target_by_group_name_and_size():
    classes, groups = {0: "pedestrian", 3: "car"}, {"person": [0, 1], "vehicle": [3, 4]}
    tracks = [[0, 0, 10, 10, 1, .9, 0], [0, 0, 50, 50, 2, .9, 3], [0, 0, 30, 30, 3, .9, 3]]
    assert dc.pick_target(tracks, "vehicle", classes, groups)[4] == 2
    assert dc.pick_target(tracks, "person", classes, groups)[4] == 1
    assert dc.pick_target(tracks, "car", classes, groups)[4] == 2
    assert dc.pick_target(tracks, "largest", classes, groups)[4] == 2
    assert dc.pick_target(tracks[:1], "vehicle", classes, groups) is None


def _rows(seg_name, vis_frac, n=100, t0=0.0):
    out = []
    for i in range(n):
        vis = i < n * vis_frac
        out.append(dict(t=t0 + i * 0.05, seg=seg_name + rp.MOVE_SUFFIX, locked_id="3", target_visible="1" if vis else "0",
                        gs="9.0", fps="20", area="1000", setpoint="1000", reid=""))
    return out


def test_report_verdicts_and_files(tmp_path):
    segs = [dict(name="A", actor="car", speed=8, run_s=5, note=""), dict(name="B", actor="car", speed=9, run_s=5, note=""),
            dict(name="C", actor="car", speed=9, run_s=5, note="")]
    rows = [dict(t=0.0, seg="A", locked_id="3", target_visible="1", gs="0")] + _rows("A", 0.95)
    rows += _rows("B", 0.3, t0=20.0)
    s = rp.summarise(rows, segs)
    v = {x["name"]: x["verdict"] for x in s["segments"]}
    assert v == {"A": "PASS", "B": "FAIL", "C": "NOT RUN"}
    out = tmp_path / "o"
    csvp = tmp_path / "r.csv"
    keys = sorted({k for r in rows for k in r})
    with open(csvp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    rp.build(str(csvp), segs, str(out))
    page = (out / "report.html").read_text()
    assert "1/3" in page and "PASS" in page and "<svg" in page
    assert json.load(open(out / "summary.json"))["passed"] == 1


FAKE_ULTRALYTICS = textwrap.dedent('''
    import numpy as np
    class _T:
        def __init__(self, a): self.a = a
        def cpu(self): return self
        def numpy(self): return self.a
    class _Boxes:
        def __init__(self):
            self.id = _T(np.array([1.0])); self.xyxy = _T(np.array([[270.0, 165.0, 370.0, 315.0]]))
            self.conf = _T(np.array([0.9])); self.cls = _T(np.array([0.0]))
    class _R:
        def __init__(self): self.boxes = _Boxes()
    class YOLO:
        def __init__(self, path): pass
        def track(self, frame, **kw): return [_R()]
''')


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def test_run_demo_drives_the_real_tracker_end_to_end(tmp_path):
    pytest.importorskip("cv2"); pytest.importorskip("flask")
    stub = tmp_path / "stub" / "ultralytics"; stub.mkdir(parents=True)
    (stub / "__init__.py").write_text(FAKE_ULTRALYTICS)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    cfg["gcs"]["port"] = _free_port()
    cfg["model"]["path"] = str(tmp_path / "best.pt"); (tmp_path / "best.pt").write_bytes(b"x")
    cfg["run_log"]["dir"] = str(tmp_path / "runs"); cfg["logging"]["file"] = str(tmp_path / "t.log")
    cfg["video"]["source"] = str(tmp_path / "missing.mp4")
    cfgp = tmp_path / "cfg.yaml"; cfgp.write_text(yaml.safe_dump(cfg))
    scen = tmp_path / "s.yaml"
    scen.write_text(yaml.safe_dump({"name": "T", "profile": "highway", "lock_timeout_s": 4, "segments": [
        {"name": "Person seg", "actor": "person", "speed": 2, "run_s": 8, "profile": "pedestrian"},
        {"name": "Car seg", "actor": "car", "speed": 5, "run_s": 8, "profile": "vehicle"}]}))
    env = dict(os.environ, PYTHONPATH=f"{tmp_path / 'stub'}{os.pathsep}{ROOT}", PYTHONUNBUFFERED="1")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "demo", "run_demo.py"), "--scenario", str(scen),
                        "--config", str(cfgp), "--out", str(tmp_path / "out"), "--test-stack", "--headless",
                        "--no-open", "--live-port", str(_free_port()), "--time-scale", "0.5",
                        "--source", str(tmp_path / "missing.mp4")],
                       env=env, cwd=ROOT, capture_output=True, text=True, timeout=150)
    assert r.returncode == 0, r.stdout + r.stderr
    out = next((tmp_path / "out").iterdir())
    summary = json.load(open(out / "summary.json"))
    verdicts = {s["name"]: s["verdict"] for s in summary["segments"]}
    assert verdicts["Person seg"] == "PASS", (verdicts, r.stdout)       # 'lock person' command found the pedestrian
    assert verdicts["Car seg"] in ("FAIL", "PARTIAL")                   # no vehicle in view: honestly not a pass
    tlog = (out / "tracker.log").read_text()
    assert "Profile switched to 'pedestrian': cap 4.0" in tlog and "Profile switched to 'vehicle': cap 16.0" in tlog
    assert (out / "report.html").exists() and (out / "run_log.csv").exists()
    assert any(f.name.startswith("annotated") for f in out.iterdir())


def test_shipped_scenarios_name_valid_profiles_with_cap_above_target_speed():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    for f in ("exhibition.yaml", "exhibition_safe.yaml"):
        doc = yaml.safe_load(open(os.path.join(ROOT, "demo", "scenarios", f)))
        for s in sp.load_segments(doc):
            prof = cfg["profiles"][s["profile"]]            # KeyError = typo
            assert prof["flight"]["max_forward_speed"] > s["speed"], (f, s["name"])   # a drone capped at the target's speed never closes the gap
