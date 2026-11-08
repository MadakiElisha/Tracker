"""Companion-computer health: CPU temperature and Raspberry Pi under-voltage / throttling.

A hot or under-powered Pi slows down; the frame rate drops and the controller sees stale targets.
PiHealth samples every few seconds in a background thread (cheap), logs a warning when something
goes wrong and exposes the latest values for the run log. On a machine without these sources
(the dev PC) every value is None and nothing is logged.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

THERMAL = Path('/sys/class/thermal/thermal_zone0/temp')

# vcgencmd get_throttled bits (current state)
UNDER_VOLTAGE_NOW, FREQ_CAPPED_NOW, THROTTLED_NOW, SOFT_TEMP_LIMIT_NOW = 0x1, 0x2, 0x4, 0x8


def read_temp_c(path: Path = THERMAL):
    try:
        return int(path.read_text().strip()) / 1000.0
    except Exception:
        return None


def read_throttled(runner=None):
    """Return the get_throttled bitmask, or None when vcgencmd is not available."""
    try:
        if runner is None:
            out = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True, text=True, timeout=2).stdout
        else:
            out = runner()
        return int(out.strip().split('=')[1], 16)
    except Exception:
        return None


def describe(bits) -> str:
    if not bits:
        return "ok"
    names = []
    if bits & UNDER_VOLTAGE_NOW:
        names.append("UNDER-VOLTAGE")
    if bits & THROTTLED_NOW:
        names.append("throttled")
    if bits & FREQ_CAPPED_NOW:
        names.append("freq-capped")
    if bits & SOFT_TEMP_LIMIT_NOW:
        names.append("soft-temp-limit")
    return ",".join(names) or "past-event"


class PiHealth:
    def __init__(self, log=None, period_s: float = 2.0, warn_temp_c: float = 80.0,
                 temp_reader=read_temp_c, throttle_reader=read_throttled):
        self.log = log
        self.period_s = period_s
        self.warn_temp_c = warn_temp_c
        self._temp_reader, self._throttle_reader = temp_reader, throttle_reader
        self.temp_c = None
        self.throttled = None
        self._warned = set()
        self._stop = threading.Event()
        self._thread = None

    def sample(self):
        self.temp_c = self._temp_reader()
        self.throttled = self._throttle_reader()
        now_bits = (self.throttled or 0) & 0xF
        problems = set()
        if self.temp_c is not None and self.temp_c >= self.warn_temp_c:
            problems.add('hot')
        if now_bits:
            problems.add('power')
        if self.log:
            for p in problems - self._warned:
                if p == 'hot':
                    self.log.warning("PI HEALTH: CPU %.0f C (>= %.0f C): check the fan/airflow; the "
                                     "frame rate will drop.", self.temp_c, self.warn_temp_c)
                else:
                    self.log.warning("PI HEALTH: %s (get_throttled=0x%x): the Pi supply is too weak "
                                     "(5 V 5 A BEC + PSU_MAX_CURRENT=5000).", describe(now_bits), self.throttled)
            for p in self._warned - problems:
                self.log.info("PI HEALTH: %s cleared.", "temperature" if p == 'hot' else "power")
        self._warned = problems
        return self.temp_c, self.throttled

    def start(self):
        if self._temp_reader() is None and self._throttle_reader() is None:
            return self                                   # not a Pi: nothing to watch
        self._thread = threading.Thread(target=self._loop, name="pi-health", daemon=True)
        self._thread.start()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.sample()
            except Exception:
                pass
            self._stop.wait(self.period_s)

    def stop(self):
        self._stop.set()

    def row(self) -> dict:
        return {'cpu_temp_c': '' if self.temp_c is None else round(self.temp_c, 1),
                'throttled': '' if self.throttled is None else hex(self.throttled)}
