import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from telemetry_history import TelemetryHistory


def test_interpolates_between_samples():
    h = TelemetryHistory()
    h.push(0.0, tilt_rad=0.0, alt=10.0)
    h.push(1.0, tilt_rad=1.0)
    assert h.at(0.25)["tilt_rad"] == pytest.approx(0.25)
    assert h.at(0.5)["alt"] == pytest.approx(10.0)


def test_clamps_outside_the_recorded_range_and_empty_is_none():
    h = TelemetryHistory()
    assert h.at(1.0) is None
    h.push(1.0, tilt_rad=0.2)
    h.push(2.0, tilt_rad=0.4)
    assert h.at(0.0)["tilt_rad"] == pytest.approx(0.2)
    assert h.at(9.0)["tilt_rad"] == pytest.approx(0.4)


def test_late_value_is_the_old_one_not_the_current_one():
    """The point of the module: a frame taken 0.2 s ago sees the lean of 0.2 s ago."""
    h = TelemetryHistory()
    for i in range(11):                         # lean ramps 0 -> 40 deg over 1 s
        h.push(i * 0.1, tilt_rad=i * 4.0)
    assert h.at(1.0)["tilt_rad"] == pytest.approx(40.0)
    assert h.at(1.0 - 0.2)["tilt_rad"] == pytest.approx(32.0)


def test_ignores_out_of_order_and_unknown_fields():
    h = TelemetryHistory()
    h.push(1.0, tilt_rad=1.0, bogus=5)
    h.push(0.5, tilt_rad=9.0)
    assert h.at(1.0) == {"tilt_rad": 1.0}


def test_fcinterface_telemetry_at_replaces_only_time_varying_fields():
    from fc_interface5 import FCInterface
    class B:
        history = TelemetryHistory()
        def get_telemetry(self): return {"armed": True, "tilt_rad": 0.9, "alt": 8.0, "vx": 20.0, "vy": 0.0, "mode": "4"}
    B.history.push(0.0, tilt_rad=0.1, alt=7.0, vx=1.0, vy=0.0)
    B.history.push(1.0, tilt_rad=0.9, alt=8.0, vx=20.0, vy=0.0)
    fc = FCInterface.__new__(FCInterface)
    fc.backend = B()
    t = fc.telemetry_at(0.0)
    assert t["tilt_rad"] == pytest.approx(0.1) and t["armed"] is True and t["mode"] == "4"
