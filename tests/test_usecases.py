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
def test_use_case_matrix(uc):
    m = u.evaluate(uc, CFG)
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


def test_fast_profiles_declare_gimbal():
    for name, p in CFG['profiles'].items():
        if p['flight']['max_forward_speed'] > 4.0:
            assert p.get('camera', {}).get('gimbal') is True, name


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
