"""The MVP ArduPilot parameter file must agree with the tracker's own config."""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM = os.path.join(ROOT, 'params', 'mvp_13in_dakeh743.param')


def load_params():
    out = {}
    for n, line in enumerate(open(PARAM), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'([A-Z][A-Z0-9_]*),\s*(-?[0-9.]+)', line)
        assert m, f"line {n}: not NAME,VALUE: {line!r}"
        assert m.group(1) not in out, f"line {n}: {m.group(1)} set twice"
        out[m.group(1)] = float(m.group(2))
    return out


P = load_params()
CFG = yaml.safe_load(open(os.path.join(ROOT, 'config.yaml')))


def test_file_parses_and_names_fit_ardupilot_limit():
    assert len(P) > 50
    assert all(len(k) <= 16 for k in P), [k for k in P if len(k) > 16]


def test_tracker_rc_channels_are_ignored_by_ardupilot():
    rc = CFG['rc']
    for key in ('lock_channel', 'release_channel', 'filter_channel'):
        ch = rc[key]
        assert P.get(f'RC{ch}_OPTION') == 0, f"RC{ch}_OPTION must be 0: the tracker reads CH{ch}"
        assert P['FLTMODE_CH'] != ch


def test_guided_is_on_the_mode_switch_and_guided_id_matches():
    assert CFG['safety']['guided_mode_id'] == 4
    assert 4 in {P[f'FLTMODE{i}'] for i in range(1, 7)}


def test_fence_is_outside_the_tracker_leash():
    s = CFG['safety']
    assert P['FENCE_ENABLE'] == 1
    # leash settings arrive with the safety PR; 150 m + 30 m are its defaults
    assert P['FENCE_RADIUS'] >= s.get('leash_m', 150.0) + s.get('leash_rtl_margin_m', 30.0)
    assert P['FENCE_ALT_MAX'] > CFG['flight']['max_altitude']


def test_guided_timeout_and_accel_match_validation():
    assert P['GUID_TIMEOUT'] <= 1.0
    assert P['WPNAV_ACCEL'] == 400                      # the offline model's max_accel (4 m/s^2)


def test_companion_link_matches_tracker_expectations():
    assert P['SERIAL2_PROTOCOL'] == 2 and P['SERIAL2_BAUD'] == 921
    assert P['SR2_POSITION'] >= 10 and P['SR2_EXTRA1'] >= 20 and P['SR2_RC_CHAN'] >= 5


def test_failsafes_present_and_no_arming_checks_skipped():
    for k in ('FS_THR_ENABLE', 'BATT_FS_LOW_ACT', 'BATT_FS_CRT_ACT', 'FS_EKF_ACTION', 'FS_CRASH_CHECK'):
        assert P[k] > 0, k
    assert 'ARMING_CHECK' not in P and 'ARMING_SKIPCHK' not in P


def test_battery_thresholds_ordered():
    assert P['MOT_BAT_VOLT_MIN'] < P['BATT_CRT_VOLT'] < P['BATT_LOW_VOLT'] < P['MOT_BAT_VOLT_MAX']
    assert P['BATT_CRT_MAH'] < P['BATT_LOW_MAH'] < P['BATT_CAPACITY']
