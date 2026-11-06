"""Highway scenario additions: pure maths, generated worlds, parameter files, launcher defaults.

None of this needs Gazebo. It also guards the original small-world setup against accidental changes.
"""
import os, re, subprocess, sys
import xml.etree.ElementTree as ET
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM = os.path.join(ROOT, "sim")
sys.path.insert(0, SIM)
import highway_paths as hp            # noqa: E402
import make_highway_world as mhw      # noqa: E402
import make_fast_quad as mfq          # noqa: E402


def read(*p):
    with open(os.path.join(ROOT, *p)) as f:
        return f.read()


def params(path):
    out = {}
    for line in read(path).splitlines():
        line = line.split("#")[0].strip()
        if line:
            k, v = [x.strip() for x in line.split(",")[:2]]
            out[k] = v
    return out


# ---- motion maths ----------------------------------------------------------------------------
def test_distance_waits_ramps_then_cruises():
    assert hp.distance_travelled(5, 20, 2.5, delay=10) == 0.0
    assert hp.distance_travelled(14, 20, 2.5, delay=10) == pytest.approx(0.5 * 2.5 * 16)       # still ramping
    d1, d2 = hp.distance_travelled(30, 20, 2.5), hp.distance_travelled(40, 20, 2.5)
    assert d2 - d1 == pytest.approx(200.0)                                                      # 20 m/s cruise


def test_speed_never_exceeds_target_speed():
    prev = 0.0
    for i in range(1, 400):
        d = hp.distance_travelled(i * 0.1, 10.0, 3.0)
        assert (d - prev) / 0.1 <= 10.0 + 1e-6
        prev = d


def test_target_stops_at_end_of_road_and_lasts_over_a_minute():
    x, _, _ = hp.pose(10_000, 20, 2.5, -1.75)
    assert x == hp.ROAD_END_X
    # the 2 km road keeps a 20 m/s car and a 10 m/s runner moving for well over 60 s
    t_car = next(t for t in range(0, 1000) if hp.pose(t, 20, 2.5, 0)[0] >= hp.ROAD_END_X)
    t_run = next(t for t in range(0, 1000) if hp.pose(t, 10, 3.0, 0)[0] >= hp.ROAD_END_X)
    assert t_car > 90 and t_run > 180


def test_lane_changes_are_smooth_and_alternate():
    ys = [hp.lane_offset(t / 10, 0, 10, 3.5) for t in range(0, 300)]
    assert max(ys) == pytest.approx(3.5) and min(ys) == pytest.approx(0.0)
    assert max(abs(b - a) for a, b in zip(ys, ys[1:])) < 0.25          # no teleporting
    assert hp.lane_offset(5, 0, 0, 3.5) == 0.0                         # disabled


# ---- worlds ----------------------------------------------------------------------------------
WORLDS = ["tracker_world.sdf", "highway_world.sdf", "highway_world_fast.sdf"]


@pytest.mark.parametrize("name", WORLDS)
def test_world_is_valid_xml_and_keeps_world_name(name):
    w = ET.parse(os.path.join(SIM, "worlds", name)).getroot().find("world")
    assert w.get("name") == "tracker_world"          # launcher wait-loop and movers depend on this
    assert w.find("physics") is not None and w.find("light") is not None


def test_highway_worlds_contain_mover_targets_and_a_drone():
    for name, drone in (("highway_world.sdf", "iris_with_ardupilot"),
                        ("highway_world_fast.sdf", "fast_quad_with_ardupilot")):
        w = ET.parse(os.path.join(SIM, "worlds", name)).getroot().find("world")
        names = {i.findtext("name") for i in w.findall("include")}
        assert {"highway_car", "highway_runner"} <= names
        assert any(i.findtext("uri") == f"model://{drone}" for i in w.findall("include"))
        assert len(list(w)) < 120                       # merged visuals: no per-dash models


@pytest.mark.parametrize("name,drone", [("highway_world.sdf", "iris_with_ardupilot"),
                                        ("highway_world_fast.sdf", "fast_quad_with_ardupilot")])
def test_committed_highway_worlds_match_generator(name, drone):
    assert read("sim", "worlds", name) == mhw.build(drone)


def test_original_world_untouched_features():
    t = read("sim", "worlds", "tracker_world.sdf")
    for n in ("moving_target", "moving_person2", "vehicle_target", "moving_suv", "model://iris_with_ardupilot"):
        assert n in t


# ---- parameters ------------------------------------------------------------------------------
def test_original_params_unchanged_in_meaning():
    p = params("sim/params/ardupilot_tracker.param")
    assert p["WPNAV_SPEED"] == "500" and p["FRAME_CLASS"] == "1" and p["FRAME_TYPE"] == "1"


def test_fast_params_raise_limits_and_never_skip_arming_checks():
    fast, base = params("sim/params/ardupilot_fast.param"), params("sim/params/ardupilot_tracker.param")
    assert int(fast["WPNAV_SPEED"]) >= 2500 > int(base["WPNAV_SPEED"])
    assert int(fast["ANGLE_MAX"]) >= 5000
    for name in ("sim/params/ardupilot_tracker.param", "sim/params/ardupilot_fast.param"):
        assert not re.search(r"^\s*ARMING_(CHECK|SKIPCHK)\b", read(name), re.M)


def test_highway_profile_cap_is_reachable_with_fast_params():
    import yaml
    cfg = yaml.safe_load(read("config.yaml"))
    fast = params("sim/params/ardupilot_fast.param")
    for prof in ("highway", "highway_runner"):
        cap = cfg["profiles"][prof]["flight"]["max_forward_speed"]
        assert cap * 100 <= int(fast["WPNAV_SPEED"])


# ---- launcher --------------------------------------------------------------------------------
def test_launcher_syntax_and_default_scene_unchanged():
    path = os.path.join(SIM, "launch_sim.sh")
    assert subprocess.run(["bash", "-n", path]).returncode == 0
    s = read("sim", "launch_sim.sh")
    assert 'WORLD_FILE="$SCRIPT_DIR/worlds/tracker_world.sdf"' in s
    assert 'PARAM_FILE="$SCRIPT_DIR/params/ardupilot_tracker.param"' in s
    assert "SCENE=${2:-default}" in s
    r = subprocess.run(["bash", path, "ardupilot", "nonsense"], capture_output=True, text=True, timeout=20)
    assert r.returncode == 1 and "unknown scene" in r.stdout


# ---- fast quad generator ---------------------------------------------------------------------
FIXTURE = """<model name="iris_with_ardupilot"><include><uri>model://iris_with_standoffs</uri></include>
<plugin name="ArduPilotPlugin"><imuName>iris_with_standoffs::imu_link::imu_sensor</imuName>
""" + "".join(f"""<control channel="{i}"><jointName>iris_with_standoffs::rotor_{i}_joint</jointName>
<multiplier>{'-' if i > 1 else ''}838</multiplier><cmd_max>2.5</cmd_max><cmd_min>-2.5</cmd_min></control>
""" for i in range(4)) + "</plugin></model>"


def test_fast_quad_transform_renames_and_scales():
    t = mfq.scale_motors(mfq.rename(FIXTURE), 1.4)
    assert "iris_with" not in t and "fast_quad_with_ardupilot" in t and "model://fast_quad<" in t
    assert t.count("<multiplier>1173.2</multiplier>") == 2 and t.count("<multiplier>-1173.2</multiplier>") == 2
    assert "<cmd_max>4.9</cmd_max>" in t and "<cmd_min>-4.9</cmd_min>" in t


def test_scale_one_is_a_no_op_on_values():
    t = mfq.scale_motors(FIXTURE, 1.0)
    assert t.count("<multiplier>838</multiplier>") == 2 and "<cmd_max>2.5</cmd_max>" in t


def test_fast_quad_rejects_wrong_model():
    with pytest.raises(ValueError):
        mfq.scale_motors("<model/>", 1.4)


def test_mover_help_runs_without_gazebo():
    r = subprocess.run([sys.executable, os.path.join(SIM, "move_highway.py"), "--help"],
                       capture_output=True, text=True, timeout=20)
    assert r.returncode == 0 and "--car-speed" in r.stdout


def test_vanish_schedule():
    assert not hp.vanished(100, 0, 0, 1.5)                       # disabled
    assert not hp.vanished(5, 0, 10, 1.5)                        # before the first hide
    hidden = [hp.vanished(t / 10, 0, 10, 1.5) for t in range(0, 400)]
    assert hidden[100] and hidden[114] and not hidden[116]       # hidden 10.0-11.5 s
    assert hidden[215] and not hidden[260]                       # next at 21.5 s (period 11.5)
    assert not hp.vanished(0.0, 20, 10, 1.5) and not hp.vanished(25, 20, 10, 1.5)   # respects start delay
