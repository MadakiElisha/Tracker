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


# ---- root cause of the first highway Gazebo run: detector dropout bursts ----
def _with_coast(coast):
    import copy
    c = copy.deepcopy(CFG)
    for p in c['profiles'].values():
        p['tracking'] = {'coast_hold_s': coast}
    return c


def test_dropout_bursts_lose_the_car_unless_forward_speed_is_held():
    uc = next(x for x in u.USE_CASES if x.name == "highway car 20 m/s")
    air = dict(burst_prob=0.01, burst_frames=8, seed=3)
    bad = u.evaluate(uc, _with_coast(0.0), airframe=air)
    good = u.evaluate(uc, CFG, airframe=air)
    assert u.verdict(bad) == "FAIL" and bad['visible'] < 0.5      # the symptom you saw: lock lost
    assert u.verdict(good) == "PASS"


def test_profile_coast_values_are_bounded_and_vehicle_stays_short():
    for name, p in CFG['profiles'].items():
        assert 0.0 <= p['tracking']['coast_hold_s'] <= 1.0, name
    assert CFG['profiles']['vehicle']['tracking']['coast_hold_s'] <= 0.25    # longer coasting loses lane changes


def test_coast_decays_yaw_and_climb_but_holds_forward_speed():
    from tracker_state import CommandState
    c = CommandState()
    c.update(10.0, 1.0, 0.5)
    c.decay_split(1.0, 0.5)
    assert c.read() == (10.0, 0.5, 0.25)


def test_fast_profiles_use_steadier_command_smoothing():
    for name in ("runner", "vehicle", "motorcycle", "highway_runner", "highway"):
        assert CFG['profiles'][name]['pid']['smooth_alpha'] >= 0.6
    assert 'smooth_alpha' not in CFG['profiles']['pedestrian']['pid']


def test_default_config_keeps_original_behaviour_switches():
    assert CFG['tracking']['coast_hold_s'] == 0.0
    assert CFG['pid']['smooth_alpha'] == 0.4
    assert CFG['tracking']['reid']['enabled'] is True


# ---- frame/telemetry time sync and delay compensation (second highway run) ----
def test_fast_chase_depends_on_tilt_being_matched_to_frame_time():
    """Documents WHY camera.sync_lag_s must be calibrated (scripts/calibrate_sync.py): an error of two
    frames (~0.1 s) in which lean is applied to the image breaks the sprinter chase outright."""
    uc = next(x for x in u.USE_CASES if x.name == "highway sprinter 10 m/s")
    assert u.verdict(u.evaluate(uc, CFG, extra=dict(tilt_lag_error_frames=0))) == "PASS"
    assert u.verdict(u.evaluate(uc, CFG, extra=dict(tilt_lag_error_frames=2))) == "FAIL"
    assert u.verdict(u.evaluate(uc, CFG, extra=dict(tilt_sync=False))) == "FAIL"     # latest lean, not frame-time lean


def test_latency_comp_is_off_by_default_and_on_for_fast_profiles():
    assert 'latency_comp' not in CFG['pid']
    for name in ("runner", "vehicle", "motorcycle", "highway_runner", "highway"):
        assert CFG['profiles'][name]['pid']['latency_comp']['seconds'] > 0
    assert CFG['camera']['sync_lag_s'] > 0


def test_range_error_prediction_shifts_toward_far_when_target_pulls_away():
    cfgp = dict(CFG['pid'], latency_comp={'seconds': 0.3}, feedforward={'enabled': True})
    pid = TrackerPID.from_config(cfgp)
    est = pid.estimator
    est.last_D, est.v_rel, est.v_sigma = 20.0, 6.0, 0.5          # target 20 m away, opening at 6 m/s
    assert pid._predict_range_error(0.10, 8.0) > 0.10
    est.v_rel = -6.0                                              # closing: predicts a smaller error
    assert pid._predict_range_error(0.10, 8.0) < 0.10
    est.v_sigma = 5.0                                             # untrusted estimate: unchanged
    assert pid._predict_range_error(0.10, 8.0) == 0.10
    off = TrackerPID.from_config(CFG['pid'])
    assert off.latency_comp_s == 0.0
