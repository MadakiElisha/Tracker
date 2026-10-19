#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Safety Supervisor

 Pure decision logic (no I/O, no threads, injectable clock) so
 every failsafe rule can be unit-tested.

 evaluate() is called once per vision-loop iteration and
 returns a SafetyDecision:

   action     NONE | HOVER | RTL | RELEASE_CONTROL
   reason     short machine-readable string
   suppress_commands  True if velocity commands must not be sent

 Rules (thresholds from config.yaml -> safety):
   * Pilot override : airborne but FC mode != GUIDED -> RELEASE_CONTROL
                      (tracker stops commanding, drops lock; resumes only
                      after the pilot returns to GUIDED and re-locks)
   * Link loss      : no FC telemetry > link_loss_warn_s  -> HOVER
                      > link_loss_abort_s -> RELEASE_CONTROL
   * Video loss     : no new frame > video_loss_hover_s   -> HOVER
                      > video_loss_rtl_s -> RTL
   * Battery        : voltage below battery_rtl_v -> RTL (0 disables)
 Highest-severity rule wins: RTL > RELEASE_CONTROL > HOVER > NONE.
=============================================================
"""

import time
from dataclasses import dataclass
from typing import Callable, Optional

NONE, HOVER, RELEASE_CONTROL, RTL = "NONE", "HOVER", "RELEASE_CONTROL", "RTL"
_SEVERITY = {NONE: 0, HOVER: 1, RELEASE_CONTROL: 2, RTL: 3}


@dataclass(frozen=True)
class SafetyDecision:
    action: str = NONE
    reason: str = ""

    @property
    def suppress_commands(self) -> bool:
        return self.action != NONE


def limit_vertical(vz: float, alt_m: float,
                   min_alt_m: float, max_alt_m: float) -> float:
    """
    Clamp a NED vertical velocity (positive = DOWN) so the tracker can never
    command the drone below min_alt_m or above max_alt_m.

    The Z loop centres the target in the frame, so without this a target near
    the ground can pull the drone down into it. At or beyond a limit only the
    direction back toward the allowed band is permitted, and the climb/descent
    is tapered over the last 2 m so it approaches the limit smoothly.
    """
    if alt_m <= 0.0:                      # altitude unknown/not reported
        return vz
    taper = 2.0
    if vz > 0.0:                          # descending
        room = alt_m - min_alt_m
        if room <= 0.0:
            return 0.0
        return vz * min(1.0, room / taper)
    if vz < 0.0:                          # climbing
        room = max_alt_m - alt_m
        if room <= 0.0:
            return 0.0
        return vz * min(1.0, room / taper)
    return vz


class SafetySupervisor:
    def __init__(self, cfg: dict,
                 clock: Callable[[], float] = time.monotonic):
        self.cfg    = cfg
        self._clock = clock
        self.guided_mode = str(cfg.get('guided_mode_id', 4))
        self._last_frame_t: Optional[float] = None

    def note_frame(self):
        """Call whenever a fresh video frame has been read."""
        self._last_frame_t = self._clock()

    def evaluate(self, *, airborne: bool, takeoff_in_progress: bool,
                 mode: str, telemetry_age_s: Optional[float],
                 battery_v: float) -> SafetyDecision:
        c = self.cfg
        best = SafetyDecision()

        def consider(action, reason):
            nonlocal best
            if _SEVERITY[action] > _SEVERITY[best.action]:
                best = SafetyDecision(action, reason)

        if not airborne:
            return best                      # nothing to protect on the ground

        # Pilot override (ignore while our own takeoff is switching modes)
        if not takeoff_in_progress and str(mode) != self.guided_mode:
            consider(RELEASE_CONTROL, f"pilot_override(mode={mode})")

        # Link loss
        if telemetry_age_s is not None:
            if telemetry_age_s > c.get('link_loss_abort_s', 5.0):
                consider(RELEASE_CONTROL, f"link_lost({telemetry_age_s:.1f}s)")
            elif telemetry_age_s > c.get('link_loss_warn_s', 1.0):
                consider(HOVER, f"link_stale({telemetry_age_s:.1f}s)")

        # Video loss
        if self._last_frame_t is not None:
            age = self._clock() - self._last_frame_t
            if age > c.get('video_loss_rtl_s', 5.0):
                consider(RTL, f"video_lost({age:.1f}s)")
            elif age > c.get('video_loss_hover_s', 0.5):
                consider(HOVER, f"video_stale({age:.1f}s)")

        # Battery (0 disables; 0.0 reading = not reported yet -> ignore)
        rtl_v = c.get('battery_rtl_v', 0.0)
        if rtl_v > 0.0 and 0.0 < battery_v < rtl_v:
            consider(RTL, f"battery_low({battery_v:.2f}V)")

        return best
