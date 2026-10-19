#!/usr/bin/env python3
"""
Drone Tracker — per-frame CSV run log.

One row per vision-loop iteration. analyze_run.py scores a run from this file,
so every sim/flight leaves evidence you can review afterwards.
"""
import csv
import time
from pathlib import Path
from typing import Optional

FIELDS = [
    "t", "frame", "fps", "airborne", "mode", "alt", "battery",
    "tracks", "locked_id", "target_visible", "lost_frames",
    "err_x", "err_y", "area", "setpoint",
    "cmd_vx", "cmd_vz", "cmd_yaw",
    "safety_action", "safety_reason", "errors",
]


class RunLogger:
    def __init__(self, directory: str, enabled: bool = True,
                 clock=time.time, flush_every: int = 10):
        self.enabled = enabled
        self._clock = clock
        self._n = 0
        self._flush_every = flush_every
        self.path: Optional[Path] = None
        self._fh = None
        self._w = None
        self._t0 = None
        if enabled:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.path = Path(directory) / time.strftime("run_%Y%m%d_%H%M%S.csv")
            self._fh = open(self.path, "w", newline="")
            self._w = csv.DictWriter(self._fh, fieldnames=FIELDS)
            self._w.writeheader()

    def row(self, **kw):
        if not self.enabled:
            return
        now = self._clock()
        self._t0 = self._t0 if self._t0 is not None else now
        rec = {k: "" for k in FIELDS}
        rec.update({k: v for k, v in kw.items() if k in FIELDS})
        rec["t"] = round(now - self._t0, 3)
        self._w.writerow(rec)
        self._n += 1
        if self._n % self._flush_every == 0:
            self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None
