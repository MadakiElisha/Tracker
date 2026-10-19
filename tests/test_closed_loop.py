"""
Closed-loop model scenarios (see closed_loop.py for what is and is not modelled).
Thresholds are set from measured results with margin, so a regression in the
controller (sign error, saturation, windup, lost altitude limit) trips them.
"""
import copy, dataclasses, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import yaml
import closed_loop as cl

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config.yaml')))['pid']

SLOW = dict(latency_frames=6, tau_v=0.8, seed=2)      # 200 ms pipeline, sluggish airframe


@pytest.mark.parametrize("variant", [{}, SLOW], ids=["nominal", "slow+laggy"])
@pytest.mark.parametrize("scn", cl.SCENARIOS, ids=[s.name for s in cl.SCENARIOS])
def test_target_stays_in_view_and_centred(scn, variant):
    m = cl.metrics(cl.run(dataclasses.replace(scn, **variant), CFG))
    assert m['visible_frac'] >= 0.99, f"target left the frame: {m}"
    assert m['median_abs_err_x'] <= 0.06, m
    assert m['yaw_flips_per_10s'] <= 2.0, f"yaw oscillating: {m}"


@pytest.mark.parametrize("variant", [{}, SLOW], ids=["nominal", "slow+laggy"])
@pytest.mark.parametrize("scn", [s for s in cl.SCENARIOS if 'stop' not in s.name],
                         ids=[s.name for s in cl.SCENARIOS if 'stop' not in s.name])
def test_holds_standoff_distance_while_target_moves(scn, variant):
    m = cl.metrics(cl.run(dataclasses.replace(scn, **variant), CFG))
    assert m['median_abs_range_err'] <= 0.20, m


def test_stop_and_go_never_loses_target_or_collides():
    """Known weakness: after a stop the drone settles slowly (median range error
    ~24%). It must still never lose the target or fly into it."""
    scn = [s for s in cl.SCENARIOS if 'stop' in s.name][0]
    r = cl.run(scn, CFG)
    m = cl.metrics(r)
    assert m['visible_frac'] >= 0.99
    assert m['median_abs_range_err'] <= 0.35
    assert min(r.dist[15 * 30:]) >= 2.0, "drone flew within 2 m of the target"


def test_altitude_floor_is_respected_everywhere():
    for scn in cl.SCENARIOS:
        r = cl.run(scn, CFG)
        assert min(r.alt) >= scn.min_alt - 0.05, f"{scn.name}: dipped to {min(r.alt):.2f} m"
        assert max(r.alt) <= scn.max_alt + 0.05


def test_without_altitude_limit_loop_dips_below_the_floor():
    """Why limit_vertical exists: the Z loop centres the target, so with the
    floor disabled it descends past the 3 m minimum (it is not bounded by
    anything else)."""
    scn = dataclasses.replace(cl.SCENARIOS[0], min_alt=-100.0)
    r = cl.run(scn, CFG)
    assert min(r.alt) < 2.8, f"expected a dip below the 3 m floor, min was {min(r.alt):.2f}"


def test_legacy_area_mode_cannot_follow_a_walker():
    """Regression evidence for the redesign: original gains/area-error lose a
    1.4 m/s walker."""
    legacy = copy.deepcopy(CFG)
    legacy.update(distance_mode='area', forward_gate={'enabled': False})
    legacy['pitch'] = dict(kp=8.0, ki=0.05, kd=1.5, output_limit=3.0, deadzone=0.015,
                           i_limit_frac=1e9)
    scn = [s for s in cl.SCENARIOS if 'walking away' in s.name][0]
    m = cl.metrics(cl.run(scn, legacy, setpoint_area=0.10))
    assert m['visible_frac'] < 0.95 or m['final_dist'] > 30, m


def test_deterministic():
    a = cl.metrics(cl.run(cl.SCENARIOS[1], CFG))
    b = cl.metrics(cl.run(cl.SCENARIOS[1], CFG))
    assert a == b
