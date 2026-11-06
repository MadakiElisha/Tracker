"""
Target ground-speed estimator (monocular, flat ground).

Where the bottom edge of the target's box sits in the image, plus the camera's depression angle
(mount pitch + body tilt) and the drone's altitude, gives the target's horizontal distance:

    delta = cam_pitch + tilt + atan((y_bottom - H/2) / f)        # angle below the horizon
    D     = altitude / tan(delta)                                # metres, horizontal

A 2-state constant-velocity Kalman filter tracks D and dD/dt. Because dD/dt = v_target - v_drone
along the line of sight, the target's own speed along it is  v_target = v_drone_fwd + dD/dt.
The controller adds that as feed-forward, so it commands the target's speed instead of waiting
for range error (and the integral term) to build up.

Assumptions (all stated, none hidden): flat ground, the box bottom touches the ground, altitude
and tilt come from telemetry, the drone yaws toward the target so range is roughly along its
forward axis. Output is gated by confidence and a deadband so a parked target gives ~0 and a
noisy or implausible estimate gives 0 (the P/I loop then behaves exactly as before).
"""
import math
from typing import Optional


class TargetMotionEstimator:
    def __init__(self, hfov_deg: float = 90.0, cam_pitch_deg: float = 20.0,
                 pixel_sigma: float = 3.0, accel_sigma: float = 3.0,
                 deadband_ms: float = 0.7, max_speed: float = 30.0,
                 min_delta_deg: float = 4.0, max_vel_sigma: float = 1.5,
                 max_gap_s: float = 0.5):
        self.hfov = math.radians(hfov_deg)
        self.cam_pitch = math.radians(cam_pitch_deg)
        self.pixel_sigma = pixel_sigma
        self.accel_sigma = accel_sigma
        self.deadband = deadband_ms
        self.max_speed = max_speed
        self.min_delta = math.radians(min_delta_deg)
        self.max_vel_sigma = max_vel_sigma
        self.max_gap = max_gap_s
        self.reset()

    def reset(self):
        self._x = None            # [D, Ddot]
        self._P = None
        self._t = None
        self.last_D: Optional[float] = None
        self.v_rel: float = 0.0
        self.v_sigma: float = float('inf')

    # ---- geometry --------------------------------------------------------------------------
    def ground_range(self, y_bottom: float, frame_w: int, frame_h: int,
                     alt_m: float, tilt_rad: float):
        """Return (D, sigma_D) or None if the geometry is unusable."""
        f = (frame_w / 2.0) / math.tan(self.hfov / 2.0)
        delta = self.cam_pitch + tilt_rad + math.atan((y_bottom - frame_h / 2.0) / f)
        if alt_m is None or alt_m < 1.0 or delta < self.min_delta or delta > math.radians(85):
            return None
        D = alt_m / math.tan(delta)
        # d D / d y_px = alt / (sin^2(delta)) / (f cos^2(angle_in_image)) ~ alt / (f sin^2(delta))
        sigma = self.pixel_sigma * alt_m / (f * math.sin(delta) ** 2)
        return D, max(sigma, 0.05)

    # ---- filter ------------------------------------------------------------------------------
    def update(self, t: float, box, frame_w: int, frame_h: int,
               alt_m: Optional[float], tilt_rad: float, v_drone_fwd: float) -> float:
        """Feed one detection; returns the target's forward speed estimate (m/s), 0 if not trusted."""
        gr = self.ground_range(box[3], frame_w, frame_h, alt_m, tilt_rad) if alt_m else None
        if gr is None:
            return 0.0
        D, sig = gr
        if self._x is None or self._t is None or (t - self._t) > self.max_gap:
            self._x = [D, 0.0]
            self._P = [[sig ** 2, 0.0], [0.0, 25.0]]
            self._t = t
            self.last_D = D
            self.v_rel, self.v_sigma = 0.0, float('inf')
            return 0.0
        dt = max(t - self._t, 1e-3)
        self._t = t
        # predict
        x0, x1 = self._x
        x0 += x1 * dt
        P = self._P
        q = self.accel_sigma ** 2
        p00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1] + q * dt ** 4 / 4
        p01 = P[0][1] + dt * P[1][1] + q * dt ** 3 / 2
        p10 = P[1][0] + dt * P[1][1] + q * dt ** 3 / 2
        p11 = P[1][1] + q * dt ** 2
        # update with measurement D
        r = sig ** 2
        innov = D - x0
        s = p00 + r
        k0, k1 = p00 / s, p10 / s
        x0 += k0 * innov
        x1 += k1 * innov
        self._P = [[(1 - k0) * p00, (1 - k0) * p01], [p10 - k1 * p00, p11 - k1 * p01]]
        self._x = [x0, x1]
        self.last_D = x0
        self.v_rel = x1
        self.v_sigma = math.sqrt(max(self._P[1][1], 0.0))
        return self.speed(v_drone_fwd)

    def speed(self, v_drone_fwd: float) -> float:
        """Target forward speed (m/s) from the latest filter state, with confidence gating + deadband."""
        if self.v_sigma > self.max_vel_sigma:
            return 0.0
        v = v_drone_fwd + self.v_rel
        if abs(v) > self.max_speed:
            return 0.0
        mag = max(0.0, abs(v) - self.deadband)       # soft threshold: parked target -> exactly 0
        return math.copysign(mag, v)
