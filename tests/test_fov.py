"""Field-of-view normalisation: tuning done on the 90 deg Gazebo camera must carry over to real lenses.

Without it, the IMX519 (66.3 deg) made the yaw loop ~53 % hotter and the drone hold people
~1.75x further away (7.6 m -> 13.3 m), and the 7 m/s sprinter case lost the target.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import usecases as u                     # noqa: E402
from pid_controller2 import TrackerPID   # noqa: E402

IMX519 = 66.3
IMX296_M12 = 83.3


def _pid(hfov, **kw):
    return TrackerPID.from_config({'camera_hfov_deg': hfov, 'setpoint_area': 0.01,
                                   'setpoint_area_by_class': {0: 0.01, 3: 0.03}, **kw})


def test_reference_camera_unchanged():
    p = _pid(90.0)
    assert p.fov_k == pytest.approx(1.0) and p.setpoint_area == pytest.approx(0.01)


def test_narrow_lens_scales_setpoints_and_bearing():
    p = _pid(IMX519)
    k = math.tan(math.radians(IMX519 / 2))
    assert p.fov_k == pytest.approx(k)
    assert p.setpoint_for(0) == pytest.approx(0.01 / k ** 2)       # same real distance -> bigger box
    assert p.setpoint_for(3) == pytest.approx(0.03 / k ** 2)


def test_normalisation_can_be_disabled():
    p = _pid(IMX519, fov_normalise=False)
    assert p.fov_k == 1.0 and p.setpoint_area == pytest.approx(0.01)


def test_same_bearing_gives_same_yaw_command_on_any_lens():
    """A target 10 deg right of centre must produce the same yaw rate on a 90 and a 66 deg lens."""
    out = []
    for hfov in (90.0, IMX519):
        p = _pid(hfov)
        f = 320 / math.tan(math.radians(hfov / 2))
        cx = 320 + f * math.tan(math.radians(10))
        out.append(p.compute((cx - 10, 200, cx + 10, 280), 640, 480, class_id=0)[2])
    assert out[0] == pytest.approx(out[1], rel=1e-6)


PEOPLE = ["pedestrian walking 1.4 m/s", "jogger 3 m/s", "sprinter 7 m/s (starts at rest)"]


@pytest.mark.parametrize("name", PEOPLE)
def test_mvp_people_cases_pass_on_imx519(name):
    cfg = u.load_cfg()
    uc = next(x for x in u.USE_CASES if x.name == name)
    ref = u.evaluate(uc, cfg)
    m = u.evaluate(uc, cfg, extra={'hfov_deg': IMX519})
    assert u.verdict(m) == "PASS", m
    assert abs(m['final_dist'] - ref['final_dist']) < 1.0, (m['final_dist'], ref['final_dist'])


@pytest.mark.parametrize("name", ["city car 12 m/s", "bus 10 m/s", "highway sprinter 10 m/s"])
def test_vehicle_speeds_need_the_wide_lens(name):
    """Documented limit: on a fixed camera these pass with the ~83 deg IMX296 M12 lens."""
    cfg = u.load_cfg()
    uc = next(x for x in u.USE_CASES if x.name == name)
    assert u.verdict(u.evaluate(uc, cfg, extra={'hfov_deg': IMX296_M12})) == "PASS"
