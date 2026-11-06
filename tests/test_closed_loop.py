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
    assert m['yaw_flips_per_10s'] <= 5.0, f"yaw oscillating: {m}"   # analyzer limit is 10


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


def _patrol(period_s):
    import math
    return lambda t: (20 * math.sin(2 * math.pi * t / period_s), -15.0)


@pytest.mark.parametrize("variant", [{}, SLOW], ids=["nominal", "slow+laggy"])
def test_walker_speed_patrol_holds_range_within_20pct(variant):
    """Regression for the first real sim log (kp 3 / ki 0.1: only 36% of frames within +-20%).
    Target patrols +-20 m at ~1.4 m/s peak, ~14 fps."""
    scn = cl.Scenario("patrol 1.4 m/s", _patrol(90), duration_s=200, class_id=3,
                      target_h=1.5, target_w=3.0, fps=14.0, latency_frames=2, seed=1)
    r = cl.run(dataclasses.replace(scn, **variant), CFG)
    ok = sum(1 for t, v, e in zip(r.t, r.visible, r.range_err) if t >= 10 and v and abs(e) <= 0.2)
    n = sum(1 for t in r.t if t >= 10)
    assert ok / n >= 0.85, f"within +-20%: {ok / n:.2f}"


def test_fast_pickup_patrol_never_lost_nominal():
    """2.1 m/s reversing target is a stress case: not required to hold +-20%, must not be lost."""
    scn = cl.Scenario("patrol 2.1 m/s", _patrol(60), duration_s=160, class_id=3,
                      target_h=1.5, target_w=3.0, fps=14.0, latency_frames=2, seed=1)
    m = cl.metrics(cl.run(scn, CFG), 10.0)
    assert m['visible_frac'] >= 0.99


def test_parked_target_with_detector_jitter_does_not_creep(monkeypatch):
    """Regression for the first stationary-target sim run: box-size jitter (~5 px) made the
    forward command swing +-0.7 m/s through the derivative term (48% of in-deadband frames
    commanded motion). Measured exactly as analyze_run.py does."""
    rec = []
    orig = cl.TrackerPID.compute

    def spy(self, box, w, h, class_id=None, **kw):
        a = ((box[2] - box[0]) * (box[3] - box[1])) / (w * h)
        out = orig(self, box, w, h, class_id, **kw)
        rec.append((((box[0] + box[2]) / 2) / w - 0.5, ((box[1] + box[3]) / 2) / h - 0.5,
                    self.distance_error(a, class_id), out))
        return out

    monkeypatch.setattr(cl.TrackerPID, "compute", spy)
    scn = cl.Scenario("parked", cl.stationary(15.0, 0.0), duration_s=120, class_id=3,
                      target_h=1.5, target_w=3.0, pixel_noise=5.5, fps=30.0,
                      latency_frames=2, seed=1)
    cl.run(scn, CFG)
    rows = rec[30 * 30:]
    still = [x for x in rows if abs(x[0]) < 0.03 and abs(x[1]) < 0.04 and abs(x[2]) < 0.05]
    assert len(still) > 100
    creeping = sum(max(abs(v) for v in x[3]) > 0.08 for x in still) / len(still)
    assert creeping < 0.05, f"{creeping:.0%} of in-deadband frames commanded motion"
