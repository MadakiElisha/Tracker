#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — 3-Axis PID Controller (config-driven)

 Takes the locked target's bounding box and outputs velocity
 commands for the FC abstraction layer.

   1. Yaw   — centres the target horizontally in frame
   2. Z     — centres the target vertically in frame
   3. Pitch — holds the target at a desired apparent size
              (area-based distance estimate)

 Conventions (MAVLink BODY_NED):
   vx   > 0  forward          error_area > 0  target too small/far
   vz   > 0  DOWN             error_y    > 0  target below centre
   yaw  > 0  clockwise/right  error_x    > 0  target right of centre

 Distance error
   mode "range" (default): e = sqrt(setpoint_area / area) - 1
        Apparent area falls as 1/distance^2, so sqrt(area) ~ 1/distance and
        e is the RELATIVE range error ((d - d_set) / d_set): positive = too
        far. It is dimensionless, independent of the object's real size, and
        does not saturate when the target is far away.
   mode "area" (legacy): e = setpoint_area - area. This saturates at
        +setpoint_area for distant targets, so forward speed is capped near
        kp * setpoint_area (0.4-0.8 m/s with the old gains) — too slow to
        follow a walking person. Kept for comparison only.

 Changes from the original:
   * gains come from config.yaml via TrackerPID.from_config()
   * dt uses an injectable clock (deterministic tests/closed-loop sims)
   * reset() restarts the clock and dt is capped, so a long gap
     between lock/compute calls no longer injects a huge integral
   * optional per-class distance setpoint
=============================================================
"""

import math
import time
from typing import Callable, Optional


def _ramp(v: float, start: float, full: float) -> float:
    """1.0 while v <= start, 0.0 once v >= full, linear in between."""
    if v <= start:
        return 1.0
    if v >= full:
        return 0.0
    return 1.0 - (v - start) / (full - start)


class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float, deadzone: float = 0.0,
                 max_dt: float = 0.2,
                 clock: Callable[[], float] = time.monotonic,
                 i_limit_frac: float = 0.3,
                 i_leak_s: float = 0.0,
                 i_flip_decay: float = 1.0):
        self.kp           = kp
        self.i_flip_decay = i_flip_decay
        self.i_leak_s     = i_leak_s
        self.i_limit_frac = i_limit_frac
        self.ki           = ki
        self.kd           = kd
        self.output_limit = output_limit
        self.deadzone     = deadzone
        self.max_dt       = max_dt
        self._clock       = clock

        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = clock()
        self._first_run   = True
        self._first_run_flag = True

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = self._clock()
        self._first_run  = True

    def compute(self, error: float) -> float:
        now = self._clock()
        dt  = min(max(now - self._prev_time, 1e-4), self.max_dt)
        self._prev_time = now

        if abs(error) < self.deadzone:
            error = 0.0

        self._first_run_flag = self._first_run
        if self._first_run:
            derivative      = 0.0           # no derivative spike on first sample
            self._first_run = False
        else:
            derivative = (error - self._prev_error) / dt

        # P and D first; the integral is added afterwards so we can apply
        # conditional integration (anti-windup).
        p_d = self.kp * error + self.kd * derivative

        # 1) Conditional integration: do not accumulate while the loop is
        #    already saturated in the direction the error is pushing. Without
        #    this a long approach winds the integral up and the vehicle
        #    overshoots the setpoint (seen in closed-loop sim).
        saturated = abs(p_d) >= self.output_limit and (p_d * error) > 0
        if (self.i_flip_decay < 1.0 and not self._first_run_flag
                and error * self._prev_error < 0):
            self._integral *= self.i_flip_decay   # error crossed zero: shed stale memory
        if self.i_leak_s > 0.0:          # leaky integrator: forget old error
            self._integral *= math.exp(-dt / self.i_leak_s)
        if not saturated:
            self._integral += error * dt
        # 2) Bound the integral's own contribution to a fraction of the output.
        if self.ki > 0.0:
            limit = (self.i_limit_frac * self.output_limit) / self.ki
            self._integral = max(-limit, min(limit, self._integral))

        output = p_d + self.ki * self._integral
        output = max(-self.output_limit, min(self.output_limit, output))

        self._prev_error = error
        return output


class TrackerPID:
    """Three PIDController loops for 3D follow-me. See module docstring."""

    def __init__(self, cfg: Optional[dict] = None,
                 clock: Callable[[], float] = time.monotonic):
        cfg = cfg or {}
        self._clock_fn = clock
        max_dt = cfg.get('max_dt', 0.2)

        def mk(name, d_kp, d_ki, d_kd, d_lim, d_dz):
            c = cfg.get(name, {})
            return PIDController(
                kp=c.get('kp', d_kp), ki=c.get('ki', d_ki),
                kd=c.get('kd', d_kd),
                output_limit=c.get('output_limit', d_lim),
                deadzone=c.get('deadzone', d_dz),
                max_dt=max_dt, clock=clock,
                i_limit_frac=c.get('i_limit_frac', 0.3),
                i_leak_s=c.get('i_leak_s', 0.0),
                i_flip_decay=c.get('i_flip_decay', 1.0))

        self.distance_mode = cfg.get('distance_mode', 'range')
        if self.distance_mode not in ('range', 'area'):
            raise ValueError(f"distance_mode must be 'range' or 'area', got {self.distance_mode!r}")
        self.max_range_error = float(cfg.get('max_range_error', 5.0))

        self.yaw_pid = mk('yaw', 1.2, 0.02, 0.15, 1.0, 0.03)
        self.z_pid   = mk('z',   1.5, 0.01, 0.20, 1.5, 0.04)
        if self.distance_mode == 'range':
            self.pitch_pid = mk('pitch', 1.5, 0.05, 0.20, 3.0, 0.05)
        else:
            self.pitch_pid = mk('pitch', 8.0, 0.05, 1.50, 3.0, 0.015)

        zc = cfg.get('z', {})
        self.z_mode = zc.get('mode', 'image')                # 'image' (legacy) | 'altitude'
        if self.z_mode not in ('image', 'altitude'):
            raise ValueError(f"pid.z.mode must be 'image' or 'altitude', got {self.z_mode!r}")
        self.z_target_alt = float(zc.get('target_altitude', 10.0))
        self.z_alt_kp = float(zc.get('altitude_kp', 0.5))     # m/s per metre of altitude error
        self.z_edge = float(zc.get('image_edge', 0.30))       # |err_y| beyond this: image keeps target in view
        self.z_out_limit = float(zc.get('output_limit', 1.5))

        self.setpoint_area = cfg.get('setpoint_area', 0.02)
        self.setpoint_by_class = {int(k): float(v) for k, v in
                                  cfg.get('setpoint_area_by_class', {}).items()}
        # Forward-speed gating ("look before you go"): while the target sits
        # near the edge of the frame, do not close distance — let the yaw/Z
        # loops centre it first, otherwise a fixed camera loses the target.
        g = cfg.get('forward_gate', {})
        self.gate_enabled = g.get('enabled', True)
        self.gate_x = (g.get('x_start', 0.20), g.get('x_full', 0.45))
        self.gate_y = (g.get('y_start', 0.25), g.get('y_full', 0.45))
        self.smooth_alpha = cfg.get('smooth_alpha', 0.4)
        self.camera_hfov_deg = float(cfg.get('camera_hfov_deg', 90.0))
        tc = cfg.get('tilt', {}) or {}
        self.tilt_vertical = bool(tc.get('vertical', True))      # tilt-correct the vertical error / forward gate
        self.tilt_estimator = bool(tc.get('estimator', True))    # tilt-correct the target-speed estimator
        ff = cfg.get('feedforward', {}) or {}
        self.ff_enabled = bool(ff.get('enabled', False))
        self.ff_gain = float(ff.get('gain', 1.0))
        lc = cfg.get('latency_comp', {}) or {}
        self.latency_comp_s = float(lc.get('seconds', 0.0))      # 0 = off (original behaviour)
        self.latency_comp_max = float(lc.get('max_shift', 0.5))  # cap on the range-error shift
        self.estimator = None
        if self.ff_enabled:
            from target_estimator import TargetMotionEstimator
            self.estimator = TargetMotionEstimator(
                hfov_deg=self.camera_hfov_deg,
                cam_pitch_deg=float(cfg.get('camera_pitch_deg', 20.0)),
                pixel_sigma=float(ff.get('pixel_sigma', 3.0)),
                accel_sigma=float(ff.get('accel_sigma', 3.0)),
                deadband_ms=float(ff.get('deadband_ms', 0.7)),
                max_speed=float(ff.get('max_speed', 30.0)),
                max_vel_sigma=float(ff.get('max_vel_sigma', 1.5)))
        rl = cfg.get('reverse_limit')            # m/s; limits backing away (None = same as output_limit)
        self.reverse_limit = None if rl is None else float(rl)
        self.max_accel_x = float(cfg.get('max_accel', 0.0))     # m/s^2 slew limit on vx (0 = off)
        self._last_cmd_t = None
        self.drift_cutoff = cfg.get('drift_cutoff', 0.05)

        self._smooth_vx  = 0.0
        self._smooth_vz  = 0.0
        self._smooth_yaw = 0.0

    @classmethod
    def from_config(cls, pid_cfg: dict,
                    clock: Callable[[], float] = time.monotonic):
        return cls(pid_cfg, clock)

    def setpoint_for(self, class_id: Optional[int] = None) -> float:
        if class_id is not None and class_id in self.setpoint_by_class:
            return self.setpoint_by_class[class_id]
        return self.setpoint_area

    def distance_error(self, current_area: float,
                       class_id: Optional[int] = None) -> float:
        """The error the pitch loop acts on (positive = too far)."""
        sp = self.setpoint_for(class_id)
        if self.distance_mode == 'area':
            return sp - current_area
        a = max(current_area, 1e-6)
        return min(math.sqrt(sp / a) - 1.0, self.max_range_error)

    def _predict_range_error(self, err: float, alt_m: float) -> float:
        """Smith-style delay compensation: the frame is `latency_comp_s` old, so the gap has already changed
        by (closing rate x delay). Shift the measured relative range error to where it is likely to be NOW.
        Uses the estimator's range and range-rate; does nothing until the estimate is trustworthy."""
        est = self.estimator
        if est is None or est.last_D is None or est.v_sigma > est.max_vel_sigma:
            return err
        R = math.hypot(est.last_D, alt_m)
        if R < 1.0:
            return err
        v_slant = est.v_rel * est.last_D / R                       # + = target pulling away
        shift = (1.0 + err) * v_slant * self.latency_comp_s / R
        shift = max(-self.latency_comp_max, min(self.latency_comp_max, shift))
        return min(err + shift, self.max_range_error)

    def compute(self, target_box, frame_w: int, frame_h: int,
                class_id: Optional[int] = None, tilt_rad: float = 0.0,
                alt_m: Optional[float] = None, v_drone_fwd: float = 0.0):
        """
        Returns (vx, vz, yaw_rate).
        vx forward m/s, vz m/s (NED: positive = DOWN), yaw_rate rad/s.

        tilt_rad: body pitch, nose-DOWN positive, from telemetry. A fixed camera leans with
        the drone, so a nose-down tilt shifts the target UP in the image; the vertical loop
        works on the tilt-compensated error so it does not chase the drone's own lean.
        """
        x1, y1, x2, y2 = target_box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = x2 - x1, y2 - y1

        error_x = (cx / frame_w) - 0.5
        error_y = (cy / frame_h) - 0.5
        current_area = (bw * bh) / (frame_w * frame_h)

        error_y_ctrl = error_y
        if tilt_rad and self.tilt_vertical:
            f_px = (frame_w / 2.0) / math.tan(math.radians(self.camera_hfov_deg) / 2.0)
            ang = math.atan(error_y * frame_h / f_px) + tilt_rad      # angle below the camera axis, nominal frame
            error_y_ctrl = f_px * math.tan(max(-1.4, min(1.4, ang))) / frame_h

        raw_yaw = self.yaw_pid.compute(error_x)
        if self.z_mode == 'altitude' and alt_m is not None:
            # hold the commanded altitude; the image only intervenes when the target nears the
            # top/bottom edge (so a fast chase does not drift down to the altitude floor)
            excess = 0.0
            if abs(error_y_ctrl) > self.z_edge:
                excess = math.copysign(abs(error_y_ctrl) - self.z_edge, error_y_ctrl)
            alt_term = max(-self.z_out_limit, min(self.z_out_limit,
                                                  self.z_alt_kp * (alt_m - self.z_target_alt)))
            raw_vz = max(-self.z_out_limit, min(self.z_out_limit,
                                                alt_term + self.z_pid.compute(excess)))
        else:
            raw_vz = self.z_pid.compute(error_y_ctrl)
        v_t = 0.0
        if self.estimator is not None:
            v_t = self.estimator.update(self._clock_fn(), target_box, frame_w, frame_h,
                                        alt_m, tilt_rad if self.tilt_estimator else 0.0, v_drone_fwd)
        dist_err = self.distance_error(current_area, class_id)
        if self.latency_comp_s > 0.0 and self.estimator is not None and alt_m:
            dist_err = self._predict_range_error(dist_err, alt_m)
        raw_vx  = self.pitch_pid.compute(dist_err)
        if self.estimator is not None:
            lim = self.pitch_pid.output_limit
            raw_vx = max(-lim, min(lim, raw_vx + self.ff_gain * v_t))
        if self.gate_enabled and raw_vx > 0.0:      # only gate approaching, never retreat
            raw_vx *= (_ramp(abs(error_x), *self.gate_x) *
                       _ramp(abs(error_y_ctrl), *self.gate_y))

        if self.reverse_limit is not None and raw_vx < -self.reverse_limit:
            raw_vx = -self.reverse_limit

        if self.max_accel_x > 0.0:
            now = self._clock_fn()
            if self._last_cmd_t is not None:
                dt = min(max(now - self._last_cmd_t, 1e-3), 0.2)
                step = self.max_accel_x * dt
                raw_vx = max(self._smooth_vx - step, min(self._smooth_vx + step, raw_vx))
            self._last_cmd_t = now

        a = self.smooth_alpha
        self._smooth_yaw = a * self._smooth_yaw + (1 - a) * raw_yaw
        self._smooth_vz  = a * self._smooth_vz  + (1 - a) * raw_vz
        self._smooth_vx  = a * self._smooth_vx  + (1 - a) * raw_vx

        if abs(self._smooth_yaw) < self.drift_cutoff: self._smooth_yaw = 0.0
        if abs(self._smooth_vz)  < self.drift_cutoff: self._smooth_vz  = 0.0
        if abs(self._smooth_vx)  < self.drift_cutoff: self._smooth_vx  = 0.0

        return self._smooth_vx, self._smooth_vz, self._smooth_yaw

    def reset(self):
        self.yaw_pid.reset()
        self.z_pid.reset()
        self.pitch_pid.reset()
        if self.estimator is not None:
            self.estimator.reset()
        self._smooth_vx = self._smooth_vz = self._smooth_yaw = 0.0
        self._last_cmd_t = None

    def get_debug_info(self, target_box, frame_w, frame_h,
                       class_id: Optional[int] = None) -> dict:
        x1, y1, x2, y2 = target_box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh = x2 - x1, y2 - y1
        sp = self.setpoint_for(class_id)
        cur = (bw * bh) / (frame_w * frame_h)
        return {
            'error_x':       round((cx / frame_w) - 0.5, 4),
            'error_y':       round((cy / frame_h) - 0.5, 4),
            'error_area':    round(sp - cur, 4),
            'error_dist':    round(self.distance_error(cur, class_id), 4),
            'current_area':  round(cur, 4),
            'setpoint_area': sp,
            'cmd_vx':        round(self._smooth_vx, 3),
            'cmd_vz':        round(self._smooth_vz, 3),
            'cmd_yaw':       round(self._smooth_yaw, 3),
        }
