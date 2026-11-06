"""Short history of drone telemetry so vision can use the state at the time the frame was captured.

The camera frame is ~0.1-0.3 s old by the time the detector has finished with it. A fast drone's lean
changes by tens of degrees in that time, so correcting the image with the CURRENT lean (instead of the lean
when the frame was taken) puts the correction in the wrong place and the chase falls apart (in the offline
model the 7 m/s sprinter case fails outright with unsynchronised tilt).
"""
import bisect
import threading
from collections import deque
from typing import Optional

FIELDS = ("tilt_rad", "alt", "vx", "vy")


class TelemetryHistory:
    def __init__(self, maxlen: int = 600):
        self._t = deque(maxlen=maxlen)
        self._rows = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._last = {}

    def push(self, t: float, **fields):
        """Record the latest known values at time t (fields not provided keep their previous value)."""
        with self._lock:
            self._last.update({k: float(v) for k, v in fields.items() if k in FIELDS or k == "heading"})
            if self._t and t < self._t[-1]:
                return
            self._t.append(t)
            self._rows.append(dict(self._last))

    def at(self, t: float) -> Optional[dict]:
        """Linear interpolation of the numeric fields at time t; clamps to the oldest/newest sample."""
        with self._lock:
            if not self._t:
                return None
            ts = list(self._t)
            rows = list(self._rows)
        if t <= ts[0]:
            return dict(rows[0])
        if t >= ts[-1]:
            return dict(rows[-1])
        i = bisect.bisect_right(ts, t)
        t0, t1, r0, r1 = ts[i - 1], ts[i], rows[i - 1], rows[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
        out = {}
        for k in set(r0) | set(r1):
            a, b = r0.get(k), r1.get(k)
            out[k] = a if b is None else b if a is None else a + w * (b - a)
        return out
