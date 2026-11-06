"""PID: config wiring, distance modes, gating, anti-windup, clock handling."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import yaml
from pid_controller2 import PIDController, TrackerPID, _ramp

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config.yaml')))['pid']
W, H = 640, 480


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def box(cx=0.5, cy=0.5, area=0.02):
    """Normalised centre + area fraction -> pixel box."""
    side = (area * W * H) ** 0.5
    x, y = cx * W, cy * H
    return (x - side / 2, y - side / 2, x + side / 2, y + side / 2)


def drive(pid, clk, b, n=30, class_id=None, dt=1 / 30):
    out = None
    for _ in range(n):
        clk.advance(dt)
        out = pid.compute(b, W, H, class_id=class_id)
    return out


# ---- config wiring --------------------------------------------------------
def test_gains_come_from_config():
    pid = TrackerPID.from_config(CFG)
    assert pid.pitch_pid.kp == CFG['pitch']['kp']
    assert pid.yaw_pid.kd == CFG['yaw']['kd']
    assert pid.z_pid.output_limit == CFG['z']['output_limit']
    assert pid.setpoint_area == CFG['setpoint_area']


def test_custom_gain_overrides_default():
    pid = TrackerPID.from_config({'yaw': {'kp': 9.9}})
    assert pid.yaw_pid.kp == 9.9


def test_every_config_pid_key_is_consumed():
    """A key in config.yaml that nothing reads is a lie to the operator."""
    pid = TrackerPID.from_config(CFG)
    assert pid.smooth_alpha == CFG['smooth_alpha']
    assert pid.drift_cutoff == CFG['drift_cutoff']
    assert pid.distance_mode == CFG['distance_mode']
    assert pid.max_range_error == CFG['max_range_error']
    assert pid.setpoint_by_class == {int(k): v for k, v in CFG['setpoint_area_by_class'].items()}
    assert pid.gate_x == (CFG['forward_gate']['x_start'], CFG['forward_gate']['x_full'])
    assert pid.pitch_pid.i_limit_frac == CFG['pitch']['i_limit_frac']
    assert pid.pitch_pid.max_dt == CFG['max_dt']


def test_invalid_distance_mode_rejected():
    with pytest.raises(ValueError):
        TrackerPID.from_config({'distance_mode': 'bogus'})


# ---- per-class setpoint -----------------------------------------------------
def test_per_class_setpoint_used_and_fallback():
    pid = TrackerPID.from_config(CFG)
    assert pid.setpoint_for(3) == CFG['setpoint_area_by_class'][3]
    assert pid.setpoint_for(99) == CFG['setpoint_area']
    assert pid.setpoint_for(None) == CFG['setpoint_area']


def test_same_box_is_too_far_for_pedestrian_but_ok_for_bus():
    clk = Clock()
    p1, p2 = TrackerPID.from_config(CFG, clk), TrackerPID.from_config(CFG, clk)
    b = box(area=0.03)                           # between pedestrian (0.01) and bus (0.05)
    vx_ped = drive(p1, clk, b, class_id=0)[0]    # larger than setpoint -> too close -> back off
    vx_bus = drive(p2, clk, b, class_id=8)[0]    # smaller than setpoint -> too far -> approach
    assert vx_ped <= 0 < vx_bus


# ---- sign conventions (NED body frame) ------------------------------------
@pytest.mark.parametrize("cx,cy,area,exp", [
    (0.8, 0.5, 0.02, ("vx0", "vz0", "yaw+")),     # target right -> yaw clockwise (+)
    (0.2, 0.5, 0.02, ("vx0", "vz0", "yaw-")),
    (0.5, 0.8, 0.02, ("vx0", "vz+", "yaw0")),     # target low -> descend (+, NED down)
    (0.5, 0.2, 0.02, ("vx0", "vz-", "yaw0")),     # target high -> climb (-)
])
def test_axis_signs(cx, cy, area, exp):
    clk = Clock()
    pid = TrackerPID.from_config(CFG, clk)
    vx, vz, yaw = drive(pid, clk, box(cx, cy, area))
    got = {"vx0": abs(vx) < 1e-9, "vz0": abs(vz) < 1e-9, "yaw0": abs(yaw) < 1e-9,
           "vz+": vz > 0, "vz-": vz < 0, "yaw+": yaw > 0, "yaw-": yaw < 0}
    for e in exp:
        assert got[e], f"{e} failed for box({cx},{cy},{area}): vx={vx} vz={vz} yaw={yaw}"


def test_far_target_approaches_close_target_backs_off():
    clk = Clock()
    p = TrackerPID.from_config(CFG, clk)
    assert drive(p, clk, box(area=0.002), class_id=0)[0] > 0
    p = TrackerPID.from_config(CFG, clk)
    assert drive(p, clk, box(area=0.08), class_id=0)[0] < 0


# ---- the saturation finding ---------------------------------------------------
def test_range_mode_does_not_saturate_for_distant_targets():
    """area-mode error is capped at +setpoint; range mode keeps growing."""
    clk = Clock()
    rng = TrackerPID.from_config(dict(CFG, distance_mode='range', forward_gate={'enabled': False}), clk)
    leg = TrackerPID.from_config(dict(CFG, distance_mode='area', forward_gate={'enabled': False}), clk)
    far = box(area=0.0005)
    assert rng.distance_error(0.0005, 0) > 3.0
    assert leg.distance_error(0.0005, 0) <= leg.setpoint_for(0)
    assert drive(rng, clk, far, class_id=0)[0] > 1.5


def test_range_error_is_relative_distance():
    pid = TrackerPID.from_config(CFG)
    sp = pid.setpoint_for(0)
    assert pid.distance_error(sp, 0) == pytest.approx(0.0)
    assert pid.distance_error(sp / 4, 0) == pytest.approx(1.0)     # twice as far
    assert pid.distance_error(sp * 4, 0) == pytest.approx(-0.5)    # half as far
    assert pid.distance_error(0.0, 0) == pytest.approx(pid.max_range_error)  # clamp


# ---- forward gating -----------------------------------------------------------
def test_ramp():
    assert _ramp(0.1, 0.2, 0.4) == 1.0
    assert _ramp(0.5, 0.2, 0.4) == 0.0
    assert _ramp(0.3, 0.2, 0.4) == pytest.approx(0.5)


def test_no_approach_while_target_at_frame_edge():
    clk = Clock()
    pid = TrackerPID.from_config(CFG, clk)
    vx = drive(pid, clk, box(cx=0.5, cy=0.95, area=0.002), class_id=0)[0]
    assert vx == 0.0, "must not close distance while the target is about to leave the frame"


def test_retreat_is_never_gated():
    clk = Clock()
    pid = TrackerPID.from_config(CFG, clk)
    vx = drive(pid, clk, box(cx=0.5, cy=0.95, area=0.1), class_id=0)[0]
    assert vx < 0


# ---- anti-windup / clock ---------------------------------------------------------
def test_integral_stops_winding_when_saturated():
    clk = Clock()
    pid = PIDController(kp=10, ki=1.0, kd=0, output_limit=1.0, clock=clk)
    for _ in range(500):
        clk.advance(0.05)
        pid.compute(1.0)                      # hard saturated
    assert pid._integral == pytest.approx(0.0, abs=0.06), "integral wound up while saturated"


def test_integral_contribution_bounded():
    clk = Clock()
    pid = PIDController(kp=0.01, ki=0.5, kd=0, output_limit=2.0, i_limit_frac=0.4, clock=clk)
    for _ in range(2000):
        clk.advance(0.05)
        out = pid.compute(1.0)
    assert out <= 0.4 * 2.0 + 0.01 + 1e-6


def test_stale_timestamp_does_not_inject_integral():
    """Previously reset() kept the old timestamp, so the first sample after a
    long gap integrated error * (huge dt)."""
    clk = Clock()
    pid = PIDController(kp=0, ki=1.0, kd=0, output_limit=10, clock=clk)
    pid.compute(1.0)
    clk.advance(120.0)                        # e.g. target unseen for two minutes
    pid.reset()
    clk.advance(0.033)
    out = pid.compute(1.0)
    assert out < 0.1, f"integral blew up after a long gap: {out}"


def test_dt_is_capped_even_without_reset():
    clk = Clock()
    pid = PIDController(kp=0, ki=1.0, kd=0, output_limit=10, max_dt=0.2, clock=clk)
    pid.compute(1.0)
    clk.advance(60.0)
    assert pid.compute(1.0) <= 0.2 + 1e-3     # one capped dt, plus the 1e-4 s first sample


def test_first_sample_has_no_derivative_spike():
    clk = Clock()
    pid = PIDController(kp=0, ki=0, kd=100, output_limit=1e6, clock=clk)
    clk.advance(0.033)
    assert pid.compute(0.5) == 0.0
