#!/usr/bin/env python3
"""
=============================================================================
 Drone Tracker — Phase 3: Robust 3D Flight Controller Abstraction Layer

 FIX (this version): wait_heartbeat() was resolving target_component to 0
 instead of 1 (MAV_COMP_ID_AUTOPILOT1). This silently broke every targeted
 command relying on exact component matching, including the
 MAV_CMD_SET_MESSAGE_INTERVAL request for RC_CHANNELS — which is why RC
 lock/release/filter events never fired despite MAVProxy 'rc' commands
 working fine on ArduCopter's own side.

 Fix: explicitly listen for a HEARTBEAT whose source component is the
 autopilot (1), instead of trusting whichever heartbeat arrives first.
 Also added COMMAND_ACK verification so failures are visible immediately
 instead of failing silently.
=============================================================================
"""

import time
import math
import queue
import threading
from abc import ABC, abstractmethod


# ============================================================================
# BASE BACKEND INTERFACE
# ============================================================================
class FCBackend(ABC):
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.cmd_queue = queue.Queue(maxsize=1)
        self.is_connected = False
        self.running = False
        self.last_cmd_time = 0.0
        self.watchdog_timeout = 0.5
        self.takeoff_in_progress = False
        self._worker_thread = None
        self._telem_thread = None

    @abstractmethod
    def _connect_impl(self) -> bool: pass
    @abstractmethod
    def _execute_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): pass
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
            self.is_connected = True
            self.running = True          # MUST be set before any thread starts
            self.last_cmd_time = time.time()
            # Start telemetry thread first — it only reads from the socket
            self._telem_thread = threading.Thread(
                target=self._stream_telemetry, daemon=True)
            self._telem_thread.start()
            # Start command worker thread second — it only writes to the socket
            self._worker_thread = threading.Thread(
                target=self._backend_worker, daemon=True)
            self._worker_thread.start()
            print("[FC_HAL] Telemetry and command threads started successfully.")
            return True
        return False

    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        if not self.is_connected or self.takeoff_in_progress:
            return
        if self.cmd_queue.full():
            try: self.cmd_queue.get_nowait()
            except queue.Empty: pass
        self.cmd_queue.put((vx, vy, vz, yaw_rate))
        self.last_cmd_time = time.time()

    def _backend_worker(self):
        current_vx, current_vy, current_vz, current_yaw_rate = 0.0, 0.0, 0.0, 0.0

        while self.running:
            if self.takeoff_in_progress:
                time.sleep(0.05)
                continue

            now = time.time()
            if now - self.last_cmd_time > self.watchdog_timeout:
                if any(v != 0.0 for v in [current_vx, current_vy, current_vz, current_yaw_rate]):
                    current_vx, current_vy, current_vz, current_yaw_rate = 0.0, 0.0, 0.0, 0.0
                    print("[WATCHDOG ALERT] Tracker stream timed out. Forcing zero-velocity 3D hover.")
                    try: self._execute_velocity(0.0, 0.0, 0.0, 0.0)
                    except Exception: pass
                else:
                    if self.get_telemetry().get("armed", False):
                        try: self._execute_velocity(0.0, 0.0, 0.0, 0.0)
                        except Exception: pass
                time.sleep(0.05)
                continue

            try:
                vx, vy, vz, yaw_rate = self.cmd_queue.get(timeout=0.05)
                current_vx, current_vy, current_vz, current_yaw_rate = vx, vy, vz, yaw_rate
                self._execute_velocity(current_vx, current_vy, current_vz, current_yaw_rate)
            except queue.Empty:
                if self.get_telemetry().get("armed", False):
                    try: self._execute_velocity(current_vx, current_vy, current_vz, current_yaw_rate)
                    except Exception: pass

    def disconnect(self):
        self.running = False
        if self._telem_thread:
            self._telem_thread.join(timeout=1.0)
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        self.is_connected = False


# ============================================================================
# ARDUPILOT BACKEND
# ============================================================================
class ArduPilotBackend(FCBackend):

    def _connect_impl(self) -> bool:
        print(f"[ARDUPILOT] Initializing pure MAVLink endpoint: {self.connection_string}")
        from pymavlink import mavutil

        if "udp:" in self.connection_string:
            self.master = mavutil.mavlink_connection(f"udpin:{self.connection_string.replace('udp:', '')}")
        elif ":" in self.connection_string and "/" not in self.connection_string:
            self.master = mavutil.mavlink_connection(f"udpin:{self.connection_string}")
        else:
            self.master = mavutil.mavlink_connection(self.connection_string, baud=115200)

        # ------------------------------------------------------------------
        # THE FIX: wait specifically for a HEARTBEAT whose SOURCE component
        # is the autopilot (1), not whichever heartbeat arrives first.
        # A naive wait_heartbeat() can lock onto a non-autopilot component
        # and silently break every component-targeted command afterward.
        # ------------------------------------------------------------------
        print("[ARDUPILOT] Awaiting autopilot heartbeat (component 1)...")
        AUTOPILOT_COMPONENT = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1  # = 1

        deadline = time.time() + 15.0
        found = False
        while time.time() < deadline:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=2.0)
            if msg is None:
                continue
            src_comp = msg.get_srcComponent()
            src_sys  = msg.get_srcSystem()
            print(f"  ...saw heartbeat from system={src_sys} component={src_comp}")
            if src_comp == AUTOPILOT_COMPONENT:
                self.target_system    = src_sys
                self.target_component = src_comp
                found = True
                break

        if not found:
            print("[ARDUPILOT][WARN] No heartbeat from component 1 seen — "
                  "falling back to mavutil's auto-resolved target, but RC "
                  "and some commands may not work correctly.")
            self.master.wait_heartbeat(timeout=10)
            self.target_system    = self.master.target_system
            self.target_component = self.master.target_component

        print(f"[ARDUPILOT] Connected to System: {self.target_system} "
              f"Component: {self.target_component}")

        # 1. Legacy fallback stream request (harmless even if deprecated)
        self.master.mav.request_data_stream_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )

        # 2. Explicitly request RC_CHANNELS (msg ID 65) at 10Hz
        self._request_message_interval(65, 100000)

        self._telemetry_data = {
            "armed": False, "mode": "UNKNOWN", "alt": 0.0,
            "vx": 0.0, "vy": 0.0, "vz": 0.0,
            "battery": 0.0, "gps_sats": 0,
        }
        self._telem_lock = threading.Lock()

        self._rc_lock     = threading.Lock()
        self._rc_lock_evt = False
        self._rc_rel_evt  = False
        self._rc_filter   = 'any'
        self._prev_ch7    = False
        self._prev_ch8    = False
        self._rc_msg_seen = False   # diagnostic flag

        # NOTE: _telem_thread is NOT started here.
        # It is started in connect() AFTER self.running = True is set.
        # Starting it here causes an immediate exit because self.running
        # is still False when the thread first checks 'while self.running'.
        return True

    def _request_message_interval(self, message_id: int, interval_us: int):
        """Send MAV_CMD_SET_MESSAGE_INTERVAL and verify with COMMAND_ACK."""
        from pymavlink import mavutil

        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            message_id, interval_us, 0, 0, 0, 0, 0
        )

        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3.0)
        if ack is None:
            print(f"[ARDUPILOT][WARN] No COMMAND_ACK received for "
                  f"SET_MESSAGE_INTERVAL(msg={message_id}). Stream may not be active.")
        elif ack.command == mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL:
            result_str = mavutil.mavlink.enums['MAV_RESULT'][ack.result].name
            print(f"[ARDUPILOT] SET_MESSAGE_INTERVAL(msg={message_id}) -> {result_str}")
        else:
            print(f"[ARDUPILOT][WARN] Unexpected ACK for different command "
                  f"(got {ack.command}, expected {mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL}).")

    def _stream_telemetry(self):
        from pymavlink import mavutil
        last_any_msg_time = time.time()
        warned_silent = False

        while self.running:
            try:
                msg = self.master.recv_match(
                    type=['HEARTBEAT', 'GLOBAL_POSITION_INT', 'SYS_STATUS',
                          'GPS_RAW_INT', 'RC_CHANNELS'],
                    blocking=True, timeout=0.1)

                if not msg:
                    # Watchdog: warn loudly if we've heard NOTHING for 3s.
                    # Most common cause on WSL2: a leftover zombie process
                    # still bound to the same UDP port, stealing packets.
                    if time.time() - last_any_msg_time > 3.0 and not warned_silent:
                        print("[ARDUPILOT][SILENCE WARNING] No MAVLink "
                              "messages received in 3s after a successful "
                              "connection. This usually means another "
                              "process is still bound to this UDP port. Run:"
                              "\n    ss -ulnp | grep <port>"
                              "\n  and kill any leftover python processes.")
                        warned_silent = True
                    continue

                last_any_msg_time = time.time()
                warned_silent = False

                msg_type = msg.get_type()

                if msg_type == 'HEARTBEAT':
                    # Only trust heartbeats from the autopilot component itself
                    if msg.get_srcComponent() != self.target_component:
                        continue
                    with self._telem_lock:
                        self._telemetry_data["armed"] = (
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                        self._telemetry_data["mode"] = str(msg.custom_mode)

                elif msg_type == 'GLOBAL_POSITION_INT':
                    with self._telem_lock:
                        self._telemetry_data["alt"] = msg.relative_alt / 1000.0
                        self._telemetry_data["vx"]  = msg.vx / 100.0
                        self._telemetry_data["vy"]  = msg.vy / 100.0
                        self._telemetry_data["vz"]  = msg.vz / 100.0

                elif msg_type == 'SYS_STATUS':
                    with self._telem_lock:
                        self._telemetry_data["battery"] = msg.voltage_battery / 1000.0

                elif msg_type == 'GPS_RAW_INT':
                    with self._telem_lock:
                        self._telemetry_data["gps_sats"] = msg.satellites_visible

                elif msg_type == 'RC_CHANNELS':
                    if not self._rc_msg_seen:
                        self._rc_msg_seen = True
                        print("[ARDUPILOT] First RC_CHANNELS message received "
                              "— RC stream is active.")
                    with self._rc_lock:
                        c7 = getattr(msg, 'chan7_raw', 0)
                        c8 = getattr(msg, 'chan8_raw', 0)
                        c9 = getattr(msg, 'chan9_raw', 0)

                        ch7_high = c7 > 1700
                        ch8_high = c8 > 1700

                        if ch7_high and not self._prev_ch7:
                            self._rc_lock_evt = True
                        if ch8_high and not self._prev_ch8:
                            self._rc_rel_evt = True

                        if c9 > 0:
                            if c9 < 1300:   self._rc_filter = 'any'
                            elif c9 > 1700: self._rc_filter = 'vehicle'
                            else:           self._rc_filter = 'person'

                        self._prev_ch7, self._prev_ch8 = ch7_high, ch8_high

            except Exception as e:
                print(f"[ARDUPILOT][TELEM ERROR] {e}")

    def get_telemetry(self) -> dict:
        with self._telem_lock:
            return dict(self._telemetry_data)

    def consume_rc_events(self) -> tuple:
        with self._rc_lock:
            l, r, f = self._rc_lock_evt, self._rc_rel_evt, self._rc_filter
            self._rc_lock_evt = False
            self._rc_rel_evt  = False
            return l, r, f

    def arm_and_takeoff(self, altitude_m: float) -> bool:
        from pymavlink import mavutil
        self.takeoff_in_progress = True

        print("[ARDUPILOT] Initializing Guided Mode switch Sequence...")
        self.master.set_mode(4)

        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        time.sleep(2.0)

        print(f"[ARDUPILOT] Executing NAV_TAKEOFF to {altitude_m}m...")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude_m)

        start_time = time.time()
        while time.time() - start_time < 25.0:
            if self.get_telemetry().get("alt", 0.0) >= (altitude_m - 0.4):
                break
            time.sleep(0.2)

        self.takeoff_in_progress = False
        return True

    def _execute_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        from pymavlink import mavutil
        self.master.mav.set_position_target_local_ned_send(
            0, self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            0b0000011111000111,
            0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)

    def land(self) -> bool:
        print("[ARDUPILOT] Executing Descent LAND Command Sequence.")
        self.master.set_mode(9)
        return True


# ============================================================================
# PX4 STUB BACKEND
# ============================================================================
class PX4Backend(FCBackend):
    def _connect_impl(self) -> bool: return True
    def _execute_velocity(self, vx, vy, vz, yaw_rate): pass
    def get_telemetry(self) -> dict: return {}
    def consume_rc_events(self) -> tuple: return False, False, 'any'
    def arm_and_takeoff(self, altitude_m: float) -> bool: return True
    def land(self) -> bool: return True


# ============================================================================
# FACTORY WRAPPER
# ============================================================================
class FCInterface:
    def __init__(self, fc_type: str, connection: str):
        backends = {'ardupilot': ArduPilotBackend, 'px4': PX4Backend}
        fc_type_lower = fc_type.lower()
        if fc_type_lower not in backends:
            raise ValueError(f"Unsupported stack '{fc_type}'. Choices: {list(backends.keys())}")
        self.backend = backends[fc_type_lower](connection)

    def connect(self) -> bool: return self.backend.connect()
    def get_telemetry(self) -> dict: return self.backend.get_telemetry()
    def consume_rc_events(self) -> tuple: return self.backend.consume_rc_events()
    def arm_and_takeoff(self, altitude_m: float) -> bool: return self.backend.arm_and_takeoff(altitude_m)
    def send_velocity(self, vx, vy, vz, yaw_rate): self.backend.send_velocity(vx, vy, vz, yaw_rate)
    def land(self) -> bool: return self.backend.land()
    def disconnect(self): self.backend.disconnect()


if __name__ == '__main__':
    fc = FCInterface(fc_type='ardupilot', connection='127.0.0.1:14550')
    fc.connect()
    print("\nMonitoring RC events for 30s. Try 'rc 7 2000' / 'rc 8 2000' / "
          "'rc 9 1000|1500|2000' in the MAVProxy console now.\n")
    t_end = time.time() + 30
    while time.time() < t_end:
        l, r, f = fc.consume_rc_events()
        if l: print("[TEST] LOCK event received.")
        if r: print("[TEST] RELEASE event received.")
        telem = fc.get_telemetry()
        print(f"  filter={f}  alt={telem.get('alt',0):.1f}  "
              f"armed={telem.get('armed')}", end='\r')
        time.sleep(0.2)
    fc.disconnect()
