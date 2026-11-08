"""Field safety added for the first hardware flights: leash, pilot-launched flights, minimum range."""
import math
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safety import (SafetySupervisor, RTL, NONE, ground_distance_m,   # noqa: E402
                    leash_limit_vx, adopt_airborne)
from pid_controller2 import TrackerPID                                  # noqa: E402
from tracker_state import TrackerState                                   # noqa: E402
import test_fc_and_thread as h                                           # noqa: E402

HOME = (9.0765, 7.3986)                       # any lat/lon works; flat-earth maths
M_LAT = 1.0 / 111194.9                        # degrees of latitude per metre


def north(m):
    return (HOME[0] + m * M_LAT, HOME[1])


# ---- geometry ---------------------------------------------------------------------------------
def test_ground_distance():
    assert ground_distance_m(*HOME, *north(100)) == pytest.approx(100, rel=1e-3)
    e = (HOME[0], HOME[1] + 100 * M_LAT / math.cos(math.radians(HOME[0])))
    assert ground_distance_m(*HOME, *e) == pytest.approx(100, rel=1e-3)


# ---- leash --------------------------------------------------------------------------------------
def test_leash_inside_passes():
    assert leash_limit_vx(3.0, 0, north(100), HOME, 150) == 3.0


def test_leash_outside_blocks_flying_further_out():
    assert leash_limit_vx(3.0, 0, north(160), HOME, 150) == 0.0       # heading north = outward
    assert leash_limit_vx(3.0, 45, north(160), HOME, 150) == 0.0      # still has an outward part


def test_leash_outside_allows_flying_home_and_backing_off():
    assert leash_limit_vx(3.0, 180, north(160), HOME, 150) == 3.0     # heading south = home
    assert leash_limit_vx(-2.0, 0, north(160), HOME, 150) == -2.0     # reversing toward home


def test_leash_off_or_no_position():
    assert leash_limit_vx(3.0, 0, north(500), HOME, 0) == 3.0
    assert leash_limit_vx(3.0, 0, None, HOME, 150) == 3.0
    assert leash_limit_vx(3.0, 0, north(500), None, 150) == 3.0


def test_supervisor_rtl_only_well_past_leash():
    sup = SafetySupervisor({'leash_rtl_margin_m': 30.0})
    kw = dict(airborne=True, takeoff_in_progress=False, mode="4", telemetry_age_s=0.1, battery_v=24.0)
    assert sup.evaluate(**kw, dist_home_m=170, leash_m=150).action == NONE
    d = sup.evaluate(**kw, dist_home_m=185, leash_m=150)
    assert d.action == RTL and d.reason.startswith("leash_exceeded")
    assert sup.evaluate(**kw, dist_home_m=5000, leash_m=0).action == NONE


# ---- pilot-launched flights ---------------------------------------------------------------------
def test_adopt_airborne_rule():
    assert adopt_airborne(True, 2.0, 1.5)
    assert not adopt_airborne(True, 1.0, 1.5)       # still on/near the ground
    assert not adopt_airborne(False, 10.0, 1.5)     # disarmed
    assert not adopt_airborne(True, 10.0, 0.0)      # disabled


def test_pilot_launched_flight_is_adopted_and_commanded_in_guided():
    fc, st = h.FakeFC(armed=True, alt=5.0, mode="4"), TrackerState()     # not airborne per tracker
    th, sup = h.run_thread(fc, st, adopt_airborne_alt_m=1.5)
    t_end = time.time() + 0.5
    while time.time() < t_end:
        st.cmd.update(1.0, 0.0, 0.0)
        sup.note_frame()
        time.sleep(0.02)
    h.stop(th, st)
    assert st.is_airborne and fc.vel and fc.vel[-1][0] == 1.0


def test_pilot_launched_flight_in_loiter_gets_no_commands():
    fc, st = h.FakeFC(armed=True, alt=5.0, mode="5"), TrackerState()     # LOITER: pilot flying
    th, sup = h.run_thread(fc, st, adopt_airborne_alt_m=1.5)
    for _ in range(20):
        st.cmd.update(1.0, 0.0, 0.0)
        sup.note_frame()
        time.sleep(0.02)
    h.stop(th, st)
    assert st.is_airborne and not fc.vel
    assert st.safety.action == "RELEASE_CONTROL"


def test_leash_in_fc_thread_stops_outward_flight():
    p = north(160)
    fc, st = h.FakeFC(lat=HOME[0], lon=HOME[1], heading=0), h.fresh_airborne()
    th, sup = h.run_thread(fc, st, leash_m=150.0, leash_rtl_margin_m=30.0)
    time.sleep(0.2)                                   # home recorded at HOME
    fc.t.update(lat=p[0], lon=p[1])                   # now 160 m north, heading north
    t_end = time.time() + 0.4
    while time.time() < t_end:
        st.cmd.update(3.0, 0.0, 0.0)
        sup.note_frame()
        time.sleep(0.02)
    h.stop(th, st)
    assert fc.vel[-1][0] == 0.0 and st.dist_home_m == pytest.approx(160, rel=0.01)
    assert "rtl" not in fc.calls


# ---- minimum range ------------------------------------------------------------------------------
def _pid(**kw):
    return TrackerPID.from_config({'setpoint_area': 0.01, 'min_range_m': 4.0, 'forward_gate': {'enabled': False},
                                   'smooth_alpha': 0.0, **kw})


def _box_at(pid, ground_m, alt=6.0, w=640, h=480, height_px=20):
    """A box whose bottom edge sits on the ground ground_m ahead (pinhole, flat ground)."""
    f = (w / 2) / math.tan(math.radians(pid.camera_hfov_deg) / 2)
    ang = math.atan(alt / ground_m) - pid.camera_pitch_rad
    yb = h / 2 + f * math.tan(ang)
    return (310, yb - height_px, 330, yb)


def test_ground_range_geometry():
    p = _pid()
    assert p.ground_range(_box_at(p, 10.0)[3], 640, 480, 6.0) == pytest.approx(10.0, rel=1e-6)


def test_min_range_blocks_approach_even_when_box_looks_far():
    """Partly hidden target: tiny box (looks far) but its feet are 3 m away -> no approach."""
    p = _pid()
    vx, _, _ = p.compute(_box_at(p, 3.0, height_px=8), 640, 480, class_id=0, alt_m=6.0)
    assert p.min_range_active and vx <= 0.0


def test_min_range_allows_approach_when_far():
    p = _pid()
    vx, _, _ = p.compute(_box_at(p, 20.0, height_px=8), 640, 480, class_id=0, alt_m=6.0)
    assert not p.min_range_active and vx > 0.0


def test_min_range_per_class_and_off():
    p = _pid(min_range_by_class={3: 8.0})
    p.compute(_box_at(p, 6.0, height_px=8), 640, 480, class_id=3, alt_m=6.0)
    assert p.min_range_active                         # car: 8 m
    p2 = _pid(min_range_m=0.0)
    p2.compute(_box_at(p2, 2.0, height_px=8), 640, 480, class_id=0, alt_m=6.0)
    assert not p2.min_range_active


def test_shipped_config_safety_defaults():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml')))
    s, p = cfg['safety'], cfg['pid']
    assert s['leash_m'] == 150.0 and s['leash_sim_m'] >= 2500.0 and s['adopt_airborne_alt_m'] == 1.5
    assert p['min_range_m'] == 4.0
