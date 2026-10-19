"""Safety supervisor and vertical limiter."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import yaml
from safety import (SafetySupervisor, limit_vertical, NONE, HOVER,
                    RELEASE_CONTROL, RTL)

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'config.yaml')))['safety']


class Clock:
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t


def sup(**over):
    clk = Clock()
    s = SafetySupervisor(dict(CFG, **over), clk)
    s.note_frame()
    return s, clk


def ev(s, **kw):
    base = dict(airborne=True, takeoff_in_progress=False, mode="4",
                telemetry_age_s=0.1, battery_v=12.0)
    base.update(kw)
    return s.evaluate(**base)


def test_nominal_is_none():
    s, _ = sup()
    d = ev(s)
    assert d.action == NONE and not d.suppress_commands


def test_nothing_triggers_on_the_ground():
    s, clk = sup(battery_rtl_v=11.0)
    clk.t += 60
    assert ev(s, airborne=False, mode="0", telemetry_age_s=99, battery_v=5).action == NONE


@pytest.mark.parametrize("mode", ["0", "2", "5", "6", "9", "16", "UNKNOWN"])
def test_pilot_override_any_non_guided_mode(mode):
    s, _ = sup()
    d = ev(s, mode=mode)
    assert d.action == RELEASE_CONTROL and d.reason.startswith("pilot_override")
    assert d.suppress_commands


def test_mode_ignored_during_own_takeoff():
    s, _ = sup()
    assert ev(s, mode="0", takeoff_in_progress=True).action == NONE


def test_link_stale_then_lost():
    s, _ = sup()
    assert ev(s, telemetry_age_s=0.9).action == NONE
    assert ev(s, telemetry_age_s=1.5).action == HOVER
    assert ev(s, telemetry_age_s=6.0).action == RELEASE_CONTROL


def test_video_stale_then_rtl():
    s, clk = sup()
    clk.t += 0.3
    assert ev(s).action == NONE
    clk.t += 0.5                      # 0.8 s since frame
    d = ev(s)
    assert d.action == HOVER and d.reason.startswith("video_stale")
    clk.t += 5                        # 5.8 s
    d = ev(s)
    assert d.action == RTL and d.reason.startswith("video_lost")


def test_note_frame_clears_video_stale():
    s, clk = sup()
    clk.t += 2
    assert ev(s).action == HOVER
    s.note_frame()
    assert ev(s).action == NONE


def test_no_video_rule_before_first_frame():
    clk = Clock()
    s = SafetySupervisor(CFG, clk)            # note_frame never called
    clk.t += 100
    assert ev(s).action == NONE


def test_battery_rtl_and_disabled_by_default():
    s, _ = sup(battery_rtl_v=10.5)
    assert ev(s, battery_v=10.0).action == RTL
    assert ev(s, battery_v=11.0).action == NONE
    assert ev(s, battery_v=0.0).action == NONE         # 0 = not reported
    s2, _ = sup(battery_rtl_v=0.0)
    assert ev(s2, battery_v=1.0).action == NONE         # disabled


def test_highest_severity_wins():
    s, clk = sup()
    clk.t += 6                                          # video lost -> RTL
    d = ev(s, mode="5", telemetry_age_s=2.0)            # + override + stale link
    assert d.action == RTL


# ---- vertical limiter ---------------------------------------------------------
def test_never_descends_below_floor():
    assert limit_vertical(1.5, alt_m=3.0, min_alt_m=3.0, max_alt_m=30) == 0.0
    assert limit_vertical(1.5, alt_m=2.0, min_alt_m=3.0, max_alt_m=30) == 0.0


def test_never_climbs_above_ceiling():
    assert limit_vertical(-1.5, alt_m=30.0, min_alt_m=3.0, max_alt_m=30) == 0.0


def test_allows_motion_back_into_band():
    assert limit_vertical(-1.0, alt_m=2.0, min_alt_m=3.0, max_alt_m=30) == -1.0   # climb OK below floor
    assert limit_vertical(1.0, alt_m=31.0, min_alt_m=3.0, max_alt_m=30) == 1.0    # descend OK above ceiling


def test_taper_near_floor_is_monotonic():
    outs = [limit_vertical(1.0, a, 3.0, 30) for a in (3.0, 3.5, 4.0, 4.5, 5.0, 8.0)]
    assert outs == sorted(outs) and outs[0] == 0.0 and outs[-1] == 1.0


def test_unknown_altitude_passes_through():
    assert limit_vertical(1.0, 0.0, 3.0, 30) == 1.0
