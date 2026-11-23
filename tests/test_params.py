"""The MVP ArduPilot parameter files (H743 and F405) must agree with the tracker's own config."""
import os
import re

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import pytest

FILES = {'h743': 'mvp_13in_dakeh743.param', 'f405': 'mvp_13in_dakef405.param'}
# per board: (ELRS serial port, companion-computer serial port)
PORTS = {'h743': (5, 2), 'f405': (2, 3)}


def load_params(name):
    out = {}
    for n, line in enumerate(open(os.path.join(ROOT, 'params', FILES[name])), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r'([A-Z][A-Z0-9_]*),\s*(-?[0-9.]+)', line)
        assert m, f"line {n}: not NAME,VALUE: {line!r}"
        assert m.group(1) not in out, f"line {n}: {m.group(1)} set twice"
        out[m.group(1)] = float(m.group(2))
    return out


ALL = {b: load_params(b) for b in FILES}
CFG = yaml.safe_load(open(os.path.join(ROOT, 'config.yaml')))


@pytest.mark.parametrize('board', sorted(FILES))
def test_file_parses_and_names_fit_ardupilot_limit(board):
    P = ALL[board]
    assert len(P) > 50
    assert all(len(k) <= 16 for k in P), [k for k in P if len(k) > 16]


@pytest.mark.parametrize('board', sorted(FILES))
def test_tracker_rc_channels_are_ignored_by_ardupilot(board):
    P = ALL[board]
    rc = CFG['rc']
    for key in ('lock_channel', 'release_channel', 'filter_channel'):
        ch = rc[key]
        assert P.get(f'RC{ch}_OPTION') == 0, f"RC{ch}_OPTION must be 0: the tracker reads CH{ch}"
        assert P['FLTMODE_CH'] != ch


@pytest.mark.parametrize('board', sorted(FILES))
def test_guided_is_on_the_mode_switch_and_guided_id_matches(board):
    P = ALL[board]
    assert CFG['safety']['guided_mode_id'] == 4
    assert 4 in {P[f'FLTMODE{i}'] for i in range(1, 7)}


@pytest.mark.parametrize('board', sorted(FILES))
def test_fence_is_outside_the_tracker_leash(board):
    P = ALL[board]
    s = CFG['safety']
    assert P['FENCE_ENABLE'] == 1
    # leash settings arrive with the safety PR; 150 m + 30 m are its defaults
    assert P['FENCE_RADIUS'] >= s.get('leash_m', 150.0) + s.get('leash_rtl_margin_m', 30.0)
    assert P['FENCE_ALT_MAX'] > CFG['flight']['max_altitude']


@pytest.mark.parametrize('board', sorted(FILES))
def test_guided_timeout_and_accel_match_validation(board):
    P = ALL[board]
    assert P['GUID_TIMEOUT'] <= 1.0
    assert P['WPNAV_ACCEL'] == 400                      # the offline model's max_accel (4 m/s^2)


@pytest.mark.parametrize('board', sorted(FILES))
def test_companion_link_matches_tracker_expectations(board):
    P = ALL[board]
    rc, link = PORTS[board]
    assert P[f'SERIAL{link}_PROTOCOL'] == 2
    assert P[f'SERIAL{link}_BAUD'] * 1000 >= CFG['fc']['baud'] * 0.99      # 921 -> 921600
    assert P[f'SR{link}_POSITION'] >= 10 and P[f'SR{link}_EXTRA1'] >= 20 and P[f'SR{link}_RC_CHAN'] >= 5
    assert P[f'SERIAL{rc}_PROTOCOL'] == 23                                  # ELRS / CRSF
    other_mavlink = [k for k, v in P.items() if re.fullmatch(r'SERIAL\d_PROTOCOL', k) and v == 2
                     and k != f'SERIAL{link}_PROTOCOL']
    assert not other_mavlink, other_mavlink


@pytest.mark.parametrize('board', sorted(FILES))
def test_failsafes_present_and_no_arming_checks_skipped(board):
    P = ALL[board]
    for k in ('FS_THR_ENABLE', 'BATT_FS_LOW_ACT', 'BATT_FS_CRT_ACT', 'FS_EKF_ACTION', 'FS_CRASH_CHECK'):
        assert P[k] > 0, k
    assert 'ARMING_CHECK' not in P and 'ARMING_SKIPCHK' not in P


@pytest.mark.parametrize('board', sorted(FILES))
def test_battery_thresholds_ordered(board):
    P = ALL[board]
    assert P['MOT_BAT_VOLT_MIN'] < P['BATT_CRT_VOLT'] < P['BATT_LOW_VOLT'] < P['MOT_BAT_VOLT_MAX']
    assert P['BATT_CRT_MAH'] < P['BATT_LOW_MAH'] < P['BATT_CAPACITY']


def test_f405_differs_only_where_the_board_does():
    """Same safety/flight settings on both boards; only ports and the notch mode may differ."""
    a, b = ALL['h743'], ALL['f405']
    allowed = {'INS_HNTCH_OPTS'} | {k for k in set(a) | set(b) if re.match(r'(SERIAL|SR)\d', k)}
    diff = sorted(k for k in set(a) | set(b) if k not in allowed and a.get(k) != b.get(k))
    assert not diff, diff
