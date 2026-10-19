#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Tracker State

 Encapsulates all shared mutable state into a single class.
 Replaces module-level globals that caused hidden coupling,
 untestability, and thread-safety ambiguity in tracker2bi.py.

 All threads receive a TrackerState instance explicitly.
 No function reaches into global scope for state.
=============================================================
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from tracker_logger import ExceptionCounter, get_logger

log = get_logger(__name__)


@dataclass
class CommandState:
    """
    Velocity commands computed by the PID controller.
    Written by the main vision loop, read by the FC thread.
    Protected by a single lock.
    """
    vx:   float = 0.0   # forward m/s
    vy:   float = 0.0   # lateral m/s (unused — no strafing)
    vz:   float = 0.0   # vertical m/s (NED: positive = DOWN)
    yaw:  float = 0.0   # rotation rad/s

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)
    _stamp: float = field(default_factory=time.monotonic, repr=False)

    def update(self, vx: float, vz: float, yaw: float):
        with self._lock:
            self.vx  = vx
            self.vz  = vz
            self.yaw = yaw
            self._stamp = time.monotonic()

    def zero(self):
        with self._lock:
            self.vx = self.vy = self.vz = self.yaw = 0.0
            self._stamp = time.monotonic()

    def decay(self, factor: float = 0.8):
        with self._lock:
            self.vx  *= factor
            self.vz  *= factor
            self.yaw *= factor
            self._stamp = time.monotonic()

    def read(self) -> tuple:
        with self._lock:
            return self.vx, self.vz, self.yaw

    def read_with_age(self) -> tuple:
        """(vx, vz, yaw, seconds since the vision loop last wrote).
        The FC thread uses the age to refuse stale commands if the
        vision loop hangs."""
        with self._lock:
            return (self.vx, self.vz, self.yaw,
                    time.monotonic() - self._stamp)


@dataclass
class TelemetryState:
    """
    Vehicle telemetry received from the FC.
    Written by the FC telemetry thread, read by draw functions.
    Protected by a single lock.
    """
    alt:      float = 0.0
    heading:  int   = 0
    mode:     str   = "UNKNOWN"
    armed:    bool  = False
    battery:  float = 0.0
    gps_sats: int   = 0
    vx:       float = 0.0
    vy:       float = 0.0
    vz:       float = 0.0

    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False)

    def update(self, data: dict):
        with self._lock:
            for k, v in data.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def snapshot(self) -> dict:
        """Return a copy safe to use outside the lock."""
        with self._lock:
            return {
                'alt':      self.alt,
                'heading':  self.heading,
                'mode':     self.mode,
                'armed':    self.armed,
                'battery':  self.battery,
                'gps_sats': self.gps_sats,
                'vx':       self.vx,
                'vy':       self.vy,
                'vz':       self.vz,
            }


class TrackerState:
    """
    Single container for all shared tracker state.

    One instance is created in main() and passed explicitly
    to every thread and drawing function. Nothing is global.

    Thread ownership:
      .cmd         → written by vision loop, read by fc_thread
      .telem       → written by fc_thread, read by vision loop
      .locked_id   → written and read by vision loop only
      .lost_frames → written and read by vision loop only
      .last_boxes  → written and read by vision loop only
      flags        → set by keyboard/RC/GCS, checked by fc_thread
    """

    def __init__(self):
        # ── Target lock ───────────────────────────────────────
        self.locked_id:    Optional[int]  = None
        self.lost_frames:  dict           = defaultdict(int)
        self.last_boxes:   dict           = {}

        # ── Inter-thread data ─────────────────────────────────
        self.cmd   = CommandState()
        self.telem = TelemetryState()

        # ── Flight flags ──────────────────────────────────────
        self.flag_takeoff = threading.Event()
        self.flag_land    = threading.Event()
        self.flag_rtl     = threading.Event()
        self.flag_quit    = threading.Event()
        self.is_airborne  = False
        self.rtl_issued   = False       # latch: RTL already commanded
        self.safety       = None        # latest SafetyDecision (fc thread writes)

        # ── Error tracking ────────────────────────────────────
        self.errors = ExceptionCounter()

        log.debug("TrackerState initialised.")

    # ── Convenience helpers ───────────────────────────────────

    def lock_target(self, track_id: int, cls_name: str,
                    source: str, pid=None):
        """Lock onto a track ID and reset the PID."""
        self.locked_id = track_id
        if pid is not None:
            pid.reset()
        log.info("[%s] Locked → ID:%d (%s)", source, track_id, cls_name)

    def release_lock(self, source: str, pid=None):
        """Release the current lock."""
        if self.locked_id is not None:
            log.info("[%s] Released lock (was ID:%d)",
                     source, self.locked_id)
        self.locked_id = None
        self.cmd.zero()
        if pid is not None:
            pid.reset()

    def update_lost_frames(self, active_ids: set,
                           max_lost: int, pid=None):
        """
        Increment lost frame counters for tracks no longer visible.
        Drop lock if the locked target has been gone too long.
        """
        # Seed every currently visible ID so it can be counted once it vanishes.
        for tid in active_ids:
            self.lost_frames[tid] = 0
        for tid in list(self.lost_frames.keys()):
            if tid not in active_ids:
                self.lost_frames[tid] += 1
                if self.lost_frames[tid] > max_lost:
                    if tid == self.locked_id:
                        log.warning(
                            "Lock lost: ID:%d gone for %d frames.",
                            tid, max_lost)
                        self.release_lock("LOST", pid)
                    self.lost_frames.pop(tid, None)
                    self.last_boxes.pop(tid, None)
            else:
                self.lost_frames[tid] = 0

    def resolve_tap(self, tap_xy: tuple,
                    tracks: list,
                    margin: int = 30) -> Optional[list]:
        """
        Find the track whose centre is nearest to a tap/click
        coordinate, only if the tap is within the bounding box
        (plus margin). Returns the track entry or None.
        """
        if not tracks or tap_xy is None:
            return None
        tx, ty = tap_xy
        best, best_d = None, float('inf')
        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            if x1-margin <= tx <= x2+margin and \
               y1-margin <= ty <= y2+margin:
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                d  = (cx - tx)**2 + (cy - ty)**2
                if d < best_d:
                    best_d = d
                    best   = t
        return best
