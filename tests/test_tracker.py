#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Unit Tests

 Tests for:
   - PIDController (individual axis)
   - TrackerPID (3-axis combined)
   - TrackerState (lock management, lost frame counting)
   - CommandState (thread-safe reads/writes)

 Run with:
   pytest tests/test_tracker.py -v
   pytest tests/test_tracker.py -v --tb=short
=============================================================
"""

import sys
import os
import time
import threading

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pid_controller2 import PIDController, TrackerPID
from tracker_state   import TrackerState, CommandState, TelemetryState


# ============================================================
# PIDController — individual axis tests
# ============================================================

class TestPIDController:

    def test_proportional_response(self):
        """P term: output should be proportional to error."""
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0,
                            output_limit=10.0)
        out = pid.compute(1.0)
        assert abs(out - 2.0) < 0.01, \
            f"P=2, error=1 → expected 2.0, got {out}"

    def test_zero_error_produces_zero_output(self):
        pid = PIDController(kp=5.0, ki=0.1, kd=0.5,
                            output_limit=10.0)
        for _ in range(10):
            out = pid.compute(0.0)
        assert out == 0.0, f"Zero error should produce zero output, got {out}"

    def test_output_clamped_to_limit(self):
        """Output must never exceed output_limit regardless of error."""
        pid = PIDController(kp=100.0, ki=0.0, kd=0.0,
                            output_limit=3.0)
        out = pid.compute(999.0)
        assert out <= 3.0, f"Output {out} exceeds limit 3.0"
        out = pid.compute(-999.0)
        assert out >= -3.0, f"Output {out} below limit -3.0"

    def test_deadzone_suppresses_small_errors(self):
        """Errors within the deadzone must produce zero output."""
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0,
                            output_limit=5.0, deadzone=0.05)
        out = pid.compute(0.04)  # inside deadzone
        assert out == 0.0, \
            f"Error 0.04 within deadzone 0.05 — expected 0, got {out}"

    def test_error_outside_deadzone_produces_output(self):
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0,
                            output_limit=5.0, deadzone=0.05)
        out = pid.compute(0.10)  # outside deadzone
        assert out > 0.0, \
            "Error 0.10 outside deadzone 0.05 — expected positive output"

    def test_derivative_suppressed_on_first_run(self):
        """No derivative spike on the very first compute call."""
        pid = PIDController(kp=0.0, ki=0.0, kd=100.0,
                            output_limit=50.0)
        out = pid.compute(1.0)
        assert out == 0.0, \
            f"Derivative should be zero on first run, got {out}"

    def test_derivative_fires_on_second_run(self):
        pid = PIDController(kp=0.0, ki=0.0, kd=1.0,
                            output_limit=50.0)
        pid.compute(0.0)           # first run — sets baseline
        time.sleep(0.01)
        out = pid.compute(1.0)     # second run — derivative fires
        assert out > 0.0, \
            f"Derivative should be positive on error increase, got {out}"

    def test_integral_accumulates(self):
        """Integral grows with persistent non-zero error."""
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            output_limit=100.0)
        out_prev = 0.0
        for _ in range(20):
            time.sleep(0.01)
            out = pid.compute(1.0)
            assert out >= out_prev, \
                "Integral output should increase with persistent error"
            out_prev = out

    def test_integral_anti_windup(self):
        """Integral must not grow unboundedly — anti-windup clamps it."""
        pid = PIDController(kp=0.0, ki=10.0, kd=0.0,
                            output_limit=5.0)
        for _ in range(100):
            time.sleep(0.005)
            out = pid.compute(1.0)
        assert abs(out) <= 5.0, \
            f"Anti-windup failed — output {out} exceeds limit 5.0"

    def test_reset_clears_state(self):
        """After reset, controller behaves as if newly created."""
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0,
                            output_limit=10.0)
        for _ in range(10):
            time.sleep(0.01)
            pid.compute(1.0)

        pid.reset()
        # First compute after reset should suppress derivative
        out = pid.compute(1.0)
        # With only P term active on first run: kp * 1.0 = 1.0
        # D is zero (first run), integral is zero (just reset)
        assert abs(out - 1.0) < 0.1, \
            f"After reset, first compute should behave fresh, got {out}"

    def test_negative_error_negative_output(self):
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0,
                            output_limit=10.0)
        out = pid.compute(-1.0)
        assert out < 0.0, \
            f"Negative error should produce negative output, got {out}"


# ============================================================
# TrackerPID — 3-axis combined tests
# ============================================================

class TestTrackerPID:

    def _make_box(self, cx_frac, cy_frac, area_frac,
                  fw=640, fh=480):
        """Helper: build (x1,y1,x2,y2) from fractional centre and area."""
        side = (area_frac * fw * fh) ** 0.5
        cx, cy = cx_frac * fw, cy_frac * fh
        x1 = int(cx - side / 2)
        y1 = int(cy - side / 2)
        x2 = int(cx + side / 2)
        y2 = int(cy + side / 2)
        return (x1, y1, x2, y2)

    def test_centred_target_produces_near_zero_yaw(self):
        """Target centred horizontally → yaw command ≈ 0."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.5, 0.5, 0.10)
        _, _, yaw = pid.compute(box, 640, 480)
        assert abs(yaw) < 0.1, \
            f"Centred target should produce near-zero yaw, got {yaw}"

    def test_right_of_centre_produces_positive_yaw(self):
        """Target right of centre → positive yaw (rotate right)."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.75, 0.5, 0.10)
        for _ in range(5):
            _, _, yaw = pid.compute(box, 640, 480)
            time.sleep(0.033)
        assert yaw > 0, \
            f"Target right of centre should produce positive yaw, got {yaw}"

    def test_left_of_centre_produces_negative_yaw(self):
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.25, 0.5, 0.10)
        for _ in range(5):
            _, _, yaw = pid.compute(box, 640, 480)
            time.sleep(0.033)
        assert yaw < 0, \
            f"Target left of centre should produce negative yaw, got {yaw}"

    def test_too_close_produces_negative_vx(self):
        """Target too large (too close) → negative vx (back off)."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.5, 0.5, 0.30)  # much larger than setpoint
        for _ in range(5):
            vx, _, _ = pid.compute(box, 640, 480)
            time.sleep(0.033)
        assert vx < 0, \
            f"Target too close should produce negative vx, got {vx}"

    def test_too_far_produces_positive_vx(self):
        """Target too small (too far) → positive vx (chase)."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.5, 0.5, 0.02)  # much smaller than setpoint
        for _ in range(5):
            vx, _, _ = pid.compute(box, 640, 480)
            time.sleep(0.033)
        assert vx > 0, \
            f"Target too far should produce positive vx, got {vx}"

    def test_target_below_centre_produces_positive_vz(self):
        """Target below frame centre → positive vz (descend in NED)."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.5, 0.75, 0.10)
        for _ in range(5):
            _, vz, _ = pid.compute(box, 640, 480)
            time.sleep(0.033)
        assert vz > 0, \
            f"Target below centre → descend (positive NED vz), got {vz}"

    def test_target_above_centre_produces_negative_vz(self):
        """Target above frame centre → negative vz (climb in NED)."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.5, 0.25, 0.10)
        for _ in range(5):
            _, vz, _ = pid.compute(box, 640, 480)
            time.sleep(0.033)
        assert vz < 0, \
            f"Target above centre → climb (negative NED vz), got {vz}"

    def test_output_limits_respected(self):
        """All outputs must stay within their configured limits."""
        pid = TrackerPID({'setpoint_area': 0.10})
        extreme_box = (0, 0, 10, 10)  # tiny box, far top-left
        for _ in range(10):
            vx, vz, yaw = pid.compute(extreme_box, 640, 480)
            time.sleep(0.033)
        assert abs(vx)  <= pid.pitch_pid.output_limit + 0.01
        assert abs(vz)  <= pid.z_pid.output_limit    + 0.01
        assert abs(yaw) <= pid.yaw_pid.output_limit  + 0.01

    def test_debug_info_keys_present(self):
        """get_debug_info must return all expected keys."""
        pid = TrackerPID({'setpoint_area': 0.10})
        box = (200, 150, 440, 330)
        info = pid.get_debug_info(box, 640, 480)
        required = {'error_x', 'error_y', 'error_area',
                    'current_area', 'setpoint_area',
                    'cmd_vx', 'cmd_vz', 'cmd_yaw'}
        missing = required - set(info.keys())
        assert not missing, f"Missing debug keys: {missing}"

    def test_reset_zeroes_smoothed_outputs(self):
        pid = TrackerPID({'setpoint_area': 0.10})
        box = self._make_box(0.8, 0.8, 0.30)
        for _ in range(10):
            pid.compute(box, 640, 480)
            time.sleep(0.033)
        pid.reset()
        assert pid._smooth_vx  == 0.0
        assert pid._smooth_vz  == 0.0
        assert pid._smooth_yaw == 0.0


# ============================================================
# TrackerState tests
# ============================================================

class TestTrackerState:

    def test_initial_state(self):
        state = TrackerState()
        assert state.locked_id is None
        assert len(state.lost_frames) == 0
        assert len(state.last_boxes)  == 0
        assert not state.is_airborne

    def test_lock_target(self):
        state = TrackerState()
        state.lock_target(42, "car", "TEST")
        assert state.locked_id == 42

    def test_release_lock(self):
        state = TrackerState()
        state.lock_target(42, "car", "TEST")
        state.release_lock("TEST")
        assert state.locked_id is None

    def test_release_when_nothing_locked_is_safe(self):
        state = TrackerState()
        state.release_lock("TEST")   # should not raise
        assert state.locked_id is None

    def test_lost_frame_counter_increments(self):
        state = TrackerState()
        state.lost_frames[5] = 0
        state.update_lost_frames(active_ids=set(), max_lost=10)
        assert state.lost_frames[5] == 1

    def test_lock_released_when_lost_exceeds_max(self):
        state = TrackerState()
        state.lock_target(5, "car", "TEST")
        state.lost_frames[5] = 0
        for _ in range(11):
            state.update_lost_frames(active_ids=set(), max_lost=10)
        assert state.locked_id is None, \
            "Lock should be released after max_lost frames"

    def test_lost_counter_resets_when_active(self):
        state = TrackerState()
        state.lost_frames[5] = 8
        state.update_lost_frames(active_ids={5}, max_lost=10)
        assert state.lost_frames[5] == 0

    def test_resolve_tap_finds_correct_track(self):
        state = TrackerState()
        tracks = [
            [100, 100, 200, 200, 1, 0.9, 3],  # car at centre (150,150)
            [400, 300, 500, 400, 2, 0.8, 0],  # pedestrian at (450,350)
        ]
        # Tap near first track
        result = state.resolve_tap((150, 150), tracks)
        assert result is not None
        assert int(result[4]) == 1

        # Tap near second track
        result = state.resolve_tap((450, 350), tracks)
        assert result is not None
        assert int(result[4]) == 2

    def test_resolve_tap_returns_none_when_far(self):
        state = TrackerState()
        tracks = [
            [100, 100, 200, 200, 1, 0.9, 3],
        ]
        # Tap completely away from any track
        result = state.resolve_tap((550, 400), tracks, margin=10)
        assert result is None


# ============================================================
# CommandState — thread safety tests
# ============================================================

class TestCommandState:

    def test_update_and_read(self):
        cmd = CommandState()
        cmd.update(vx=1.5, vz=-0.3, yaw=0.8)
        vx, vz, yaw = cmd.read()
        assert vx  == 1.5
        assert vz  == -0.3
        assert yaw == 0.8

    def test_zero_resets_all(self):
        cmd = CommandState()
        cmd.update(1.0, -0.5, 0.7)
        cmd.zero()
        vx, vz, yaw = cmd.read()
        assert vx == vz == yaw == 0.0

    def test_decay_reduces_magnitude(self):
        cmd = CommandState()
        cmd.update(1.0, 1.0, 1.0)
        cmd.decay(0.8)
        vx, vz, yaw = cmd.read()
        assert abs(vx  - 0.8) < 0.001
        assert abs(vz  - 0.8) < 0.001
        assert abs(yaw - 0.8) < 0.001

    def test_concurrent_write_read_is_safe(self):
        """Concurrent access must not corrupt values."""
        cmd = CommandState()
        errors = []

        def writer():
            for i in range(1000):
                try:
                    cmd.update(float(i), float(-i), float(i) * 0.1)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(1000):
                try:
                    vx, vz, yaw = cmd.read()
                    assert isinstance(vx, float)
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        assert not errors, f"Thread safety errors: {errors}"


# ============================================================
# TelemetryState tests
# ============================================================

class TestTelemetryState:

    def test_update_and_snapshot(self):
        t = TelemetryState()
        t.update({'alt': 12.5, 'mode': 'GUIDED', 'armed': True})
        snap = t.snapshot()
        assert snap['alt']   == 12.5
        assert snap['mode']  == 'GUIDED'
        assert snap['armed'] == True

    def test_unknown_keys_ignored(self):
        t = TelemetryState()
        t.update({'alt': 5.0, 'nonexistent_key': 'ignored'})
        snap = t.snapshot()
        assert snap['alt'] == 5.0
        assert 'nonexistent_key' not in snap

    def test_snapshot_is_a_copy(self):
        """Modifying the snapshot must not affect the state."""
        t = TelemetryState()
        t.update({'alt': 5.0})
        snap = t.snapshot()
        snap['alt'] = 999.0
        assert t.alt == 5.0, "Snapshot modification leaked into state"


if __name__ == '__main__':
    # Allow running directly without pytest
    import unittest
    unittest.main(verbosity=2)


def test_lock_dropped_when_target_vanishes_from_empty_start():
    """Regression: counters must be seeded by sightings, not by hand."""
    st = TrackerState()
    st.locked_id = 7
    st.update_lost_frames(active_ids={7}, max_lost=3)       # seen once
    for _ in range(5):
        st.update_lost_frames(active_ids=set(), max_lost=3)  # then gone
    assert st.locked_id is None, "lock should drop after max_lost missing frames"
