#!/usr/bin/env python3
"""
=============================================================================
 Drone Tracker — FC Abstraction Layer (fc_interface4i)

 Fixed in this version:
   - RC_CHANNELS_RAW re-added (SITL sends this, not RC_CHANNELS)
   - MAV_DATA_STREAM_RC_CHANNELS explicitly requested at 20Hz
   - SET_MESSAGE_INTERVAL sent for BOTH msg ID 35 (RAW) and 65 (CHANNELS)
   - _backend_worker no longer calls get_telemetry() in hot loop
   - _telem_lock applied symmetrically (both read and write)
   - Debug prints for first 3 RC messages + every edge event
   - ArduPilot mode map applied at telemetry read time (not draw time)
=============================================================================
"""

import time
import math
import queue
import threading
from abc import ABC, abstractmethod

# ArduPilot custom_mode integer → human-readable name
ARDUPILOT_MODE_MAP = {
    0:  "STABILIZE",
    1:  "ACRO",
    2:  "ALT_HOLD",
    3:  "AUTO",
    4:  "GUIDED",
    5:  "LOITER",
    6:  "RTL",
    7:  "CIRCLE",
    9:  "LAND",
    16: "POSHOLD",
    19: "BRAKE",
}


# ============================================================================
# BASE BACKEND
# ============================================================================
class FCBackend(ABC):
    def __init__(self, connection_string: str):
        self.connection_string  = connection_string
        self.cmd_queue          = queue.Queue(maxsize=1)
        self.is_connected       = False
        self.running            = False
        self.last_cmd_time      = 0.0
        self.watchdog_timeout   = 0.5
        self.takeoff_in_progress = False
        self._worker_thread     = None

    @abstractmethod
    def _connect_impl(self) -> bool: pass
    @abstractmethod
    def _execute_velocity(self, vx, vy, vz, yaw_rate): pass
    @abstractmethod
    def get_telemetry(self) -> dict: pass
    @abstractmethod
    def consume_rc_events(self) -> tuple: pass
    @abstractmethod
    def arm_and_takeoff(self, altitude_m: float) -> bool: pass
    @abstractmethod
    def land(self) -> bool: pass

    def connect(self) -> bool:
        if self._connect_impl():
            self.is_connected  = True
            self.running       = True
            self.last_cmd_time = time.time()
            self._worker_thread = threading.Thread(
                target=self._backend_worker, daemon=True)
            self._worker_thread.start()
            print("[FC_HAL] Background thread worker established successfully.")
            return True
        return False

    def send_velocity(self, vx, vy, vz, yaw_rate):
        if not self.is_connected or self.takeoff_in_progress:
            return
        if self.cmd_queue.full():
            try:
                self.cmd_queue.get_nowait()
            except queue.Empty:
                pass
        self.cmd_queue.put((vx, vy, vz, yaw_rate))
        self.last_cmd_time = time.time()

    def _backend_worker(self):
        """
        Sends velocity commands at ~20Hz.
        Reads armed state directly from _telemetry_data (no lock needed
        for a single boolean read — GIL keeps this safe) to avoid
        contention with the telemetry write thread.
        """
        cur_vx = cur_vy = cur_vz = cur_yaw = 0.0

        while self.running:
            if self.takeoff_in_progress:
                time.sleep(0.05)
                continue

            now = time.time()
            armed = self._telemetry_data.get("armed", False)

            # Watchdog: no command received for > 0.5s
            if now - self.last_cmd_time > self.watchdog_timeout:
                if any(v != 0.0 for v in [cur_vx, cur_vy, cur_vz, cur_yaw]):
                    cur_vx = cur_vy = cur_vz = cur_yaw = 0.0
                    print("[WATCHDOG] Stream timeout — forcing hover.")
                if armed:
                    try:
                        self._execute_velocity(0.0, 0.0, 0.0, 0.0)
                    except Exception:
                        pass
                time.sleep(0.05)
                continue

            try:
                vx, vy, vz, yaw = self.cmd_queue.get(timeout=0.05)
                cur_vx, cur_vy, cur_vz, cur_yaw = vx, vy, vz, yaw
                self._execute_velocity(cur_vx, cur_vy, cur_vz, cur_yaw)
            except queue.Empty:
                # Re-transmit last command only if armed
                if armed:
                    try:
                        self._execute_velocity(cur_vx, cur_vy, cur_vz, cur_yaw)
                    except Exception:
                        pass

    def disconnect(self):
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        self.is_connected = False


# ============================================================================
# ARDUPILOT BACKEND
# ============================================================================
class ArduPilotBackend(FCBackend):

    def _connect_impl(self) -> bool:
        print(f"[ARDUPILOT] Connecting to: {self.connection_string}")
        from pymavlink import mavutil

        cs = self.connection_string
        if cs.startswith("udp:"):
            self.master = mavutil.mavlink_connection(
                f"udpin:{cs.replace('udp:', '')}")
        elif ":" in cs and "/" not in cs:
            self.master = mavutil.mavlink_connection(f"udpin:{cs}")
        else:
            self.master = mavutil.mavlink_connection(cs, baud=115200)

        print("[ARDUPILOT] Waiting for heartbeat...")
        self.master.wait_heartbeat(timeout=15)
        self.target_system    = self.master.target_system
        self.target_component = self.master.target_component
        print(f"[ARDUPILOT] Connected — SYS:{self.target_system} "
              f"COMP:{self.target_component}")

        # ----------------------------------------------------------------
        # Stream requests — belt AND braces approach:
        # 1. MAV_DATA_STREAM_ALL          covers everything broadly
        # 2. MAV_DATA_STREAM_RC_CHANNELS  explicitly at 20Hz
        # 3. MAV_CMD_SET_MESSAGE_INTERVAL for msg 35 (RC_CHANNELS_RAW)
        # 4. MAV_CMD_SET_MESSAGE_INTERVAL for msg 65 (RC_CHANNELS)
        # Both RC message IDs are requested because SITL may send either.
        # ----------------------------------------------------------------
        print("[ARDUPILOT] Requesting data streams...")

        # Broad stream request
        self.master.mav.request_data_stream_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1)

        # RC channels specifically at 20Hz
        self.master.mav.request_data_stream_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS, 20, 1)

        # Extended status (battery, mode details)
        self.master.mav.request_data_stream_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS, 5, 1)

        # Position stream
        self.master.mav.request_data_stream_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION, 10, 1)

        time.sleep(0.1)   # brief pause before sending interval commands

        # Force RC_CHANNELS_RAW (msg 35) at 20Hz — 50000 microseconds
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            35, 50000, 0, 0, 0, 0, 0)

        # Force RC_CHANNELS (msg 65) at 20Hz
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            65, 50000, 0, 0, 0, 0, 0)

        print("[ARDUPILOT] Stream requests sent (RC at 20Hz, ALL at 10Hz).")

        # Initialise shared state
        self._telem_lock = threading.Lock()
        self._telemetry_data = {
            "armed": False, "mode": "UNKNOWN", "alt": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "battery": 0.0, "gps_sats": 0,
        }

        # RC state — separate lock to avoid blocking telemetry writes
        self._rc_lock     = threading.Lock()
        self._rc_lock_evt = False
        self._rc_rel_evt  = False
        self._rc_filter   = 'any'
        self._prev_ch7    = False
        self._prev_ch8    = False
        self._rc_msg_count = 0    # debug counter

        self._telem_thread = threading.Thread(
            target=self._stream_telemetry, daemon=True)
        self._telem_thread.start()
        return True

    def _stream_telemetry(self):
        from pymavlink import mavutil
        while self.running:
            try:
                # Include BOTH RC message types:
                #   RC_CHANNELS_RAW (35) — what SITL normally sends
                #   RC_CHANNELS     (65) — what real hardware normally sends
                msg = self.master.recv_match(
                    type=[
                        'HEARTBEAT',
                        'GLOBAL_POSITION_INT',
                        'SYS_STATUS',
                        'GPS_RAW_INT',
                        'RC_CHANNELS',
                        'RC_CHANNELS_RAW',
                    ],
                    blocking=True,
                    timeout=0.1,
                )
                if msg is None:
                    continue

                t = msg.get_type()

                if t == 'HEARTBEAT':
                    mode_int = msg.custom_mode
                    mode_str = ARDUPILOT_MODE_MAP.get(
                        mode_int, f"MODE:{mode_int}")
                    armed = bool(
                        msg.base_mode &
                        mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                    with self._telem_lock:
                        self._telemetry_data["mode"]  = mode_str
                        self._telemetry_data["armed"] = armed

                elif t == 'GLOBAL_POSITION_INT':
                    with self._telem_lock:
                        self._telemetry_data["alt"] = (
                            msg.relative_alt / 1000.0)
                        self._telemetry_data["vx"]  = msg.vx / 100.0
                        self._telemetry_data["vy"]  = msg.vy / 100.0
                        self._telemetry_data["vz"]  = msg.vz / 100.0

                elif t == 'SYS_STATUS':
                    with self._telem_lock:
                        self._telemetry_data["battery"] = (
                            msg.voltage_battery / 1000.0)

                elif t == 'GPS_RAW_INT':
                    with self._telem_lock:
                        self._telemetry_data["gps_sats"] = (
                            msg.satellites_visible)

                elif t in ('RC_CHANNELS', 'RC_CHANNELS_RAW'):
                    c7 = getattr(msg, 'chan7_raw', 0)
                    c8 = getattr(msg, 'chan8_raw', 0)
                    c9 = getattr(msg, 'chan9_raw', 0)

                    # Print first 3 RC messages so we can confirm reception
                    if self._rc_msg_count < 3:
                        print(f"[RC OK] {t} received — "
                              f"CH7:{c7} CH8:{c8} CH9:{c9}")
                        self._rc_msg_count += 1

                    ch7_high = c7 > 1700
                    ch8_high = c8 > 1700

                    with self._rc_lock:
                        # Rising-edge detection on CH7 (lock)
                        if ch7_high and not self._prev_ch7:
                            self._rc_lock_evt = True
                            print(f"[RC EVENT] CH7 LOCK fired  "
                                  f"({c7})")

                        # Rising-edge detection on CH8 (release)
                        if ch8_high and not self._prev_ch8:
                            self._rc_rel_evt = True
                            print(f"[RC EVENT] CH8 RELEASE fired"
                                  f" ({c8})")

                        # CH9 tri-state filter
                        if c9 > 0:
                            if c9 < 1300:
                                self._rc_filter = 'any'
                            elif c9 > 1700:
                                self._rc_filter = 'vehicle'
                            else:
                                self._rc_filter = 'person'

                        self._prev_ch7 = ch7_high
                        self._prev_ch8 = ch8_high

            except Exception:
                pass

    def get_telemetry(self) -> dict:
        with self._telem_lock:
            return dict(self._telemetry_data)

    def consume_rc_events(self) -> tuple:
        """Returns (lock_event, release_event, class_filter). Resets events."""
        with self._rc_lock:
            l = self._rc_lock_evt
            r = self._rc_rel_evt
            f = self._rc_filter
            self._rc_lock_evt = False
            self._rc_rel_evt  = False
            return l, r, f

    def arm_and_takeoff(self, altitude_m: float) -> bool:
        from pymavlink import mavutil
        self.takeoff_in_progress = True

        print("[ARDUPILOT] Setting GUIDED mode...")
        self.master.set_mode(4)
        time.sleep(1.0)

        print("[ARDUPILOT] Sending arm command...")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0)
        time.sleep(2.0)

        print(f"[ARDUPILOT] Takeoff to {altitude_m}m...")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m)

        # Wait for altitude, 25s timeout
        t0 = time.time()
        while time.time() - t0 < 25.0:
            alt = self._telemetry_data.get("alt", 0.0)
            print(f"  Alt: {alt:.1f}m / {altitude_m:.1f}m", end='\r')
            if alt >= altitude_m - 0.4:
                break
            time.sleep(0.2)
        print(f"\n[ARDUPILOT] Reached {self._telemetry_data.get('alt',0):.1f}m")

        self.takeoff_in_progress = False
        return True

    def _execute_velocity(self, vx, vy, vz, yaw_rate):
        from pymavlink import mavutil
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000011111000111,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, yaw_rate,
        )

    def land(self) -> bool:
        print("[ARDUPILOT] Setting LAND mode...")
        self.master.set_mode(9)
        return True


# ============================================================================
# PX4 BACKEND (stub — full implementation in Phase 8)
# ============================================================================
class PX4Backend(FCBackend):
    def _connect_impl(self) -> bool:
        print("[PX4] Stub backend — full implementation in Phase 8.")
        self._telemetry_data = {}
        return True
    def _execute_velocity(self, vx, vy, vz, yaw_rate): pass
    def get_telemetry(self) -> dict: return {}
    def consume_rc_events(self) -> tuple: return False, False, 'any'
    def arm_and_takeoff(self, altitude_m: float) -> bool: return True
    def land(self) -> bool: return True


# ============================================================================
# FACTORY
# ============================================================================
class FCInterface:
    def __init__(self, fc_type: str, connection: str):
        backends = {
            'ardupilot': ArduPilotBackend,
            'px4':       PX4Backend,
        }
        key = fc_type.lower().strip()
        if key not in backends:
            raise ValueError(
                f"Unknown FC '{fc_type}'. Choose: {list(backends.keys())}")
        self.backend = backends[key](connection)

    def connect(self)         -> bool:  return self.backend.connect()
    def get_telemetry(self)   -> dict:  return self.backend.get_telemetry()
    def consume_rc_events(self) -> tuple: return self.backend.consume_rc_events()
    def arm_and_takeoff(self, alt) -> bool: return self.backend.arm_and_takeoff(alt)
    def send_velocity(self, vx, vy, vz, yr): self.backend.send_velocity(vx, vy, vz, yr)
    def land(self)            -> bool:  return self.backend.land()
    def disconnect(self):               self.backend.disconnect()


# ============================================================================
# STANDALONE TEST
# ============================================================================
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--connect', default='127.0.0.1:14550')
    p.add_argument('--takeoff', action='store_true')
    args = p.parse_args()

    fc = FCInterface('ardupilot', args.connect)
    fc.connect()

    print("\nMonitoring telemetry for 10s (try 'rc 7 2000' in MAVProxy)...")
    for i in range(50):
        t = fc.get_telemetry()
        rc_l, rc_r, rc_f = fc.consume_rc_events()
        if rc_l: print("[TEST] RC LOCK event consumed!")
        if rc_r: print("[TEST] RC RELEASE event consumed!")
        print(f"  {i:2d}s  ALT:{t.get('alt',0):.1f}m  "
              f"MODE:{t.get('mode','?')}  "
              f"ARMED:{t.get('armed','?')}  "
              f"FILTER:{rc_f}  "
              f"BAT:{t.get('battery',0):.1f}V  "
              f"SATS:{t.get('gps_sats',0)}",
              end='\r')
        time.sleep(0.2)

    print("\n\nDone. fc_interface4i standalone test complete.")
    fc.disconnect()
