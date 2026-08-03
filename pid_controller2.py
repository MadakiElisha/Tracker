#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — 3D PID Controller

 Takes locked target state from the perception pipeline and
 outputs velocity commands for the FC abstraction layer.

 Three independent PID loops:
   1. Yaw   — centres target horizontally in frame
   2. Z     — centres target vertically in frame (altitude)
   3. Pitch — maintains target at desired apparent size
              (area-based distance estimation)
=============================================================
"""

import time

class PIDController:
    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float, deadzone: float = 0.0):
        self.kp           = kp
        self.ki           = ki
        self.kd           = kd
        self.output_limit = output_limit
        self.deadzone     = deadzone

        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = time.time()
        self._first_run   = True

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._first_run  = True

    def compute(self, error: float) -> float:
        now = time.time()
        dt  = now - self._prev_time
        dt  = max(dt, 1e-4)   # guard against zero dt
        self._prev_time = now

        # Apply deadzone
        if abs(error) < self.deadzone:
            error = 0.0

        # Derivative (skip on first run to avoid spike)
        if self._first_run:
            derivative       = 0.0
            self._first_run  = False
        else:
            derivative = (error - self._prev_error) / dt

        # Integral with anti-windup clamp
        self._integral += error * dt
        integral_limit  = self.output_limit / max(self.ki, 1e-6)
        self._integral  = max(-integral_limit, min(integral_limit, self._integral))

        # PID sum
        output = (self.kp * error +
                  self.ki * self._integral +
                  self.kd * derivative)

        # Clamp output
        output = max(-self.output_limit, min(self.output_limit, output))

        self._prev_error = error
        return output


class TrackerPID:
    """
    Wraps three PIDController instances for a 3D drone follow-me scenario.

    Frame coordinates (Normalised):
      error_x    : horizontal offset [-0.5, 0.5], positive = target right
      error_y    : vertical offset [-0.5, 0.5], positive = target below centre
      error_area : difference between setpoint area and current area
                   positive = target too far (move forward)
    """

    def __init__(self):
        # --- Yaw PID (left/right centering) ---
        self.yaw_pid = PIDController(
            kp           = 1.2,
            ki           = 0.02,
            kd           = 0.15,
            output_limit = 1.0,      # max rad/s yaw rate
            deadzone     = 0.03,     # ignore tiny horizontal errors
        )

        # --- Z PID (Altitude climb/descent centering) ---
        # Tune: Given the thrust-to-weight ratio of a 10-inch quad, keep Z-axis limits conservative
        self.z_pid = PIDController(
            kp           = 1.5,
            ki           = 0.01,
            kd           = 0.2,
            output_limit = 1.5,      # max m/s climb/descent rate
            deadzone     = 0.04,     # ignore tiny vertical errors
        )

        # --- Pitch PID (forward/backward distance hold) ---
        self.pitch_pid = PIDController(
            kp           = 8.0,
            ki           = 0.05,
            kd           = 1.5,
            output_limit = 3.0,      # max m/s forward speed
            deadzone     = 0.015,    # ignore tiny area errors
        )

        # Target apparent size in frame (fraction of frame area)
        self.setpoint_area = 0.10

        # Exponential smoothing on output commands
        self.smooth_alpha  = 0.4
        self._smooth_vx    = 0.0
        self._smooth_vz    = 0.0
        self._smooth_yaw   = 0.0
        
        # Hard cutoff to prevent micro-drift coasting
        self.drift_cutoff  = 0.05 

    def compute(self, target_box, frame_w: int, frame_h: int):
        """
        Compute velocity commands from a target bounding box.

        Returns: (vx, vz, yaw_rate)
          vx       : forward speed m/s
          vz       : vertical speed m/s (MAVLink NED: positive is DOWN)
          yaw_rate : rotation rate rad/s
        """
        x1, y1, x2, y2 = target_box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1

        # Normalised planar errors: 0 = centred, ±0.5 = at edge
        error_x = (cx / frame_w) - 0.5
        error_y = (cy / frame_h) - 0.5 

        # Area-based distance error
        current_area = (bw * bh) / (frame_w * frame_h)
        error_area   = self.setpoint_area - current_area

        # Compute raw PID outputs
        raw_yaw = self.yaw_pid.compute(error_x)
        raw_vz  = self.z_pid.compute(error_y)
        raw_vx  = self.pitch_pid.compute(error_area)

        # Exponential smoothing (Low-pass filter)
        self._smooth_yaw = (self.smooth_alpha * self._smooth_yaw + (1 - self.smooth_alpha) * raw_yaw)
        self._smooth_vz  = (self.smooth_alpha * self._smooth_vz  + (1 - self.smooth_alpha) * raw_vz)
        self._smooth_vx  = (self.smooth_alpha * self._smooth_vx  + (1 - self.smooth_alpha) * raw_vx)

        # Drift Cutoff (Eliminate micro-velocities when target is static)
        if abs(self._smooth_yaw) < self.drift_cutoff: self._smooth_yaw = 0.0
        if abs(self._smooth_vz) < self.drift_cutoff:  self._smooth_vz = 0.0
        if abs(self._smooth_vx) < self.drift_cutoff:  self._smooth_vx = 0.0

        return self._smooth_vx, self._smooth_vz, self._smooth_yaw

    def reset(self):
        self.yaw_pid.reset()
        self.z_pid.reset()
        self.pitch_pid.reset()
        self._smooth_vx  = 0.0
        self._smooth_vz  = 0.0
        self._smooth_yaw = 0.0

    def get_debug_info(self, target_box, frame_w, frame_h) -> dict:
        """Return internal state for HUD display."""
        x1, y1, x2, y2 = target_box
        cx, cy       = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        bw, bh       = x2 - x1, y2 - y1
        
        error_x      = (cx / frame_w) - 0.5
        error_y      = (cy / frame_h) - 0.5
        current_area = (bw * bh) / (frame_w * frame_h)
        error_area   = self.setpoint_area - current_area

        return {
            'error_x':      round(error_x, 4),
            'error_y':      round(error_y, 4),
            'error_area':   round(error_area, 4),
            'current_area': round(current_area, 4),
            'setpoint_area':self.setpoint_area,
            'cmd_vx':       round(self._smooth_vx, 3),
            'cmd_vz':       round(self._smooth_vz, 3),
            'cmd_yaw':      round(self._smooth_yaw, 3),
        }

if __name__ == '__main__':
    # Unit test — simulate target right of centre and low in the frame
    pid = TrackerPID()
    test_box = (420, 300, 620, 460) 
    
    for i in range(10):
        vx, vz, yaw = pid.compute(test_box, 640, 480)
        dbg = pid.get_debug_info(test_box, 640, 480)
        print(f"t={i} vx={vx:+.3f} vz={vz:+.3f} yaw={yaw:+.3f} | "
              f"errX={dbg['error_x']:+.2f} errY={dbg['error_y']:+.2f} area={dbg['current_area']:.3f}")
        time.sleep(0.033)
