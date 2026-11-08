import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_health import PiHealth, read_throttled, describe   # noqa: E402


class Log:
    def __init__(self): self.w, self.i = [], []
    def warning(self, *a): self.w.append(a[0] % a[1:])
    def info(self, *a): self.i.append(a[0] % a[1:])


def test_parse_throttled():
    assert read_throttled(lambda: "throttled=0x50005\n") == 0x50005
    assert read_throttled(lambda: "garbage") is None
    assert describe(0x5) == "UNDER-VOLTAGE,throttled" and describe(0) == "ok"


def test_warns_once_and_clears():
    temps = iter([70.0, 85.0, 86.0, 60.0])
    log = Log()
    h = PiHealth(log=log, temp_reader=lambda: next(temps), throttle_reader=lambda: 0)
    for _ in range(4):
        h.sample()
    assert len(log.w) == 1 and "CPU 85" in log.w[0]
    assert any("temperature cleared" in m for m in log.i)


def test_under_voltage_warning_and_row():
    log = Log()
    h = PiHealth(log=log, temp_reader=lambda: 55.0, throttle_reader=lambda: 0x50005)
    h.sample()
    assert any("UNDER-VOLTAGE" in m for m in log.w)
    assert h.row() == {'cpu_temp_c': 55.0, 'throttled': '0x50005'}


def test_past_events_only_do_not_warn():
    log = Log()
    h = PiHealth(log=log, temp_reader=lambda: 50.0, throttle_reader=lambda: 0x50000)
    h.sample()
    assert not log.w


def test_not_a_pi_is_silent():
    h = PiHealth(log=Log(), temp_reader=lambda: None, throttle_reader=lambda: None).start()
    assert h._thread is None and h.row() == {'cpu_temp_c': '', 'throttled': ''}
