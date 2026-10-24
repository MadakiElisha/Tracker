"""Regression tests for mission profiles, the use-case matrix, tilt compensation, altitude hold and the estimator."""
import math, os, sys
import pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import usecases as u
from profiles import apply_profile
from pid_controller2 import TrackerPID
from target_estimator import TargetMotionEstimator

CFG = u.load_cfg()
MUST_PASS = [uc for uc in u.USE_CASES if uc.expect == "pass"]


@pytest.mark.parametrize("uc", MUST_PASS, ids=lambda uc: uc.name)
def test_use_case_matrix_with_gimbal(uc):
    m = u.evaluate(uc, CFG, gimbal=True)
    assert u.verdict(m) == "PASS", m


def test_highway_case_documented_as_beyond_airframe():
    uc = next(x for x in u.USE_CASES if x.expect == "exceeds_airframe")
    assert u.verdict(u.evaluate(uc, CFG)) == "FAIL"      # if this starts passing, update the docs


def test_every_use_case_names_an_existing_profile():
    for uc in u.USE_CASES:
        assert uc.profile in CFG['profiles']


def test_profile_merges_and_does_not_mutate_base():
    base_cap = CFG['flight']['max_forward_speed']
    c = apply_profile(CFG, 'vehicle')
    assert c['flight']['max_forward_speed'] == 16.0 and c['active_profile'] == 'vehicle'
    assert c['pid']['pitch']['kp'] == 4.0 and 'profiles' not in c
    assert CFG['flight']['max_forward_speed'] == base_cap and 'active_profile' not in CFG


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="available"):
        apply_profile(CFG, 'warp')


def test_profiles_assume_fixed_camera_by_default():
    for name, p in CFG['profiles'].items():
        assert p.get('camera', {}).get('gimbal', False) is False, name


def test_altitude_mode_holds_altitude_when_target_centred():
    pid = TrackerPID.from_config(dict(CFG['pid'], z={'mode': 'altitude', 'target_altitude': 10.0}))
    box = (300, 220, 340, 260)
    _, vz_low, _ = pid.compute(box, 640, 480, class_id=3, alt_m=6.0)
    pid.reset()
    _, vz_high, _ = pid.compute(box, 640, 480, class_id=3, alt_m=14.0)
    assert vz_low < 0 < vz_high            # NED: negative = climb when low, positive = descend when high


def test_tilt_compensation_changes_vertical_error():
    pid = TrackerPID.from_config(dict(CFG['pid'], camera_hfov_deg=90.0))
    box = (300, 330, 340, 370)      # low in frame so the vertical error is non-zero
    vz0 = pid.compute(box, 640, 480, class_id=3, tilt_rad=0.0)[1]
    pid.reset()
    vz1 = pid.compute(box, 640, 480, class_id=3, tilt_rad=math.radians(30))[1]
    assert vz0 != vz1


def test_estimator_untrusted_without_altitude():
    est = TargetMotionEstimator(hfov_deg=90.0, cam_pitch_deg=45.0)
    assert est.speed(0.0) == 0.0                       # no data yet -> untrusted -> 0
    est.reset()
    assert est.update(0.0, (300, 220, 340, 260), 640, 480, alt_m=None, tilt_rad=0.0, v_drone_fwd=5.0) == 0.0


# ---- fixed camera (no gimbal): the tilt-corrected gate removed a speed oscillation ----
FIXED_CAM_MUST_PASS = ["pedestrian walking", "jogger", "sprinter", "cyclist", "motorcycle 15",
                       "city car 12 m/s", "lane changes", "cornering", "bus", "pickup patrol",
                       "highway sprinter 10", "highway car 20"]


@pytest.mark.parametrize("key", FIXED_CAM_MUST_PASS)
def test_fixed_camera_use_cases(key):
    uc = next(x for x in u.USE_CASES if key in x.name)
    m = u.evaluate(uc, CFG, gimbal=False)
    assert u.verdict(m) == "PASS", m


def test_forward_gate_uses_tilt_corrected_vertical_error():
    """A lean that shifts the target in the image must not close the forward gate when the
    tilt-corrected position is fine (otherwise speed oscillates)."""
    pid = TrackerPID.from_config(CFG['pid'])
    # small, far target (area below setpoint -> wants to advance), high in the frame
    box = (310, 40, 330, 70)
    vx_flat = pid.compute(box, 640, 480, class_id=3, tilt_rad=0.0)[0]
    pid.reset()
    vx_leaned = pid.compute(box, 640, 480, class_id=3, tilt_rad=math.radians(35))[0]
    assert vx_leaned > vx_flat       # compensation re-centres the vertical error and opens the gate


# ---- class voting and reverse limit (found in the first vehicle Gazebo run) ----
def test_class_vote_ignores_brief_flips_and_resets_on_new_lock():
    from class_vote import ClassVote
    v = ClassVote(window=30)
    out = [v.update(7, c) for c in [3] * 10 + [5, 5] + [3] * 5 + [4]]
    assert set(out) == {3}
    assert v.update(8, 5) == 5                 # a new lock id starts over


def test_class_vote_follows_a_real_change():
    from class_vote import ClassVote
    v = ClassVote(window=20)
    for _ in range(5): v.update(1, 3)
    last = [v.update(1, 8) for _ in range(30)][-1]
    assert last == 8


def test_reverse_limit_caps_backing_away_only():
    big = (200, 100, 440, 380)                 # far too close

    def worst(limit):
        cfg = dict(CFG['pid'], reverse_limit=limit)
        cfg['pitch'] = dict(cfg['pitch'], output_limit=16.0)
        pid = TrackerPID.from_config(cfg)
        return min(pid.compute(big, 640, 480, class_id=3)[0] for _ in range(30))

    assert -2.0 - 1e-9 <= worst(2.0) < 0
    assert worst(None) < -2.0


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_highway_car_robust_across_noise_seeds(seed):
    uc = next(x for x in u.USE_CASES if x.name == "highway car 20 m/s")
    m = u.evaluate(uc, CFG, gimbal=False, airframe=dict(seed=seed))
    assert u.verdict(m) == "PASS", m


def test_highway_cases_run_longer_than_a_minute_after_settling():
    for key in ("highway sprinter 10", "highway car 20"):
        uc = next(x for x in u.USE_CASES if key in x.name)
        assert uc.duration - 15.0 >= 60.0
