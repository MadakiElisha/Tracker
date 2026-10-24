import math, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import calibrate_sync as cs
from run_logger import FIELDS


def make_rows(true_lag, seconds=60, fps=19, noise_px=2.0, lean_amp=15.0, seed=1):
    rng = random.Random(seed)
    f, H = 320.0, 480
    lean = lambda t: math.radians(lean_amp) * (math.sin(2 * math.pi * 0.35 * t) + 0.5 * math.sin(2 * math.pi * 0.9 * t + 1))
    rows = []
    for k in range(int(seconds * fps)):
        t = k / fps + rng.uniform(-0.005, 0.005)
        elev = math.radians(5) * math.sin(2 * math.pi * t / 40)           # target elevation drifts slowly
        # image angle below the axis: elevation minus the lean at the moment the frame was taken
        ang = elev - lean(t - true_lag)
        err_y = (f * math.tan(ang) + rng.gauss(0, noise_px)) / H
        r = {k2: "" for k2 in FIELDS}
        r.update(t_read=round(t, 4), tilt_now_deg=round(math.degrees(lean(t)), 3), err_y=err_y,
                 airborne=1, locked_id=1, target_visible=1)
        rows.append(r)
    return rows


@pytest.mark.parametrize("lag", [0.0, 0.08, 0.17, 0.30])
def test_recovers_known_lag(lag):
    res = cs.estimate(make_rows(lag))
    assert abs(res["lag_s"] - lag) <= 0.03, res
    assert res["improvement"] > 0.6


def test_refuses_without_lean_or_samples():
    assert "barely leaned" in cs.estimate(make_rows(0.1, lean_amp=0.2))["error"]
    assert "not enough" in cs.estimate(make_rows(0.1, seconds=2))["error"]
    assert "not enough" in cs.estimate([])["error"]
