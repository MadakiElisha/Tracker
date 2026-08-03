#!/usr/bin/env python3
"""
=============================================================================
 Drone Tracker — Phase 3: Robust 3D Flight Controller Abstraction Layer
 
 Unified Architecture: Handles 4-axis velocity routing, live telemetry 
 parsing, and RC switch debouncing on a single thread-safe connection.
=============================================================================
"""

import time
import math
import queue
import threading
from abc import ABC, abstractmethod

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
            self.running = True
            self.last_cmd_time = time.time()
            self._worker_thread = threading.Thread(target=self._backend_worker, daemon=True)
            self._worker_thread.start()
            print(f"[FC_HAL] Background thread worker established successfully.")
            return True
        return False

    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        if not self.is_connected or self.takeoff_in_progress: return
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
                time.sleep(0.05)
                continue

            try:
                vx, vy, vz, yaw_rate = self.cmd_queue.get(timeout=0.05)
                current_vx, current_vy, current_vz, current_yaw_rate = vx, vy, vz, yaw_rate
                self._execute_velocity(current_vx, current_vy, current_vz, current_yaw_rate)
            except queue.Empty:
                try: self._execute_velocity(current_vx, current_vy, current_vz, current_yaw_rate)
                except Exception: pass

    def disconnect(self):
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        self.is_connected = False


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
            
        print("[ARDUPILOT] Awaiting target heartbeat verification...")
        self.master.wait_heartbeat(timeout=10)
        self.target_system = self.master.target_system
        self.target_component = self.master.target_component
        print(f"[ARDUPILOT] Connected to System: {self.target_system} Component: {self.target_component}")
        
        # Explicitly request data streams (Fix for pure pymavlink SITL environments)
        self.master.mav.request_data_stream_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_ALL, 10, 1
        )
        
        self._telemetry_data = {"armed": False, "mode": "UNKNOWN", "alt": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0, "battery": 0.0, "gps_sats": 0}
        
        # Thread-safe internal RC event buffers
        self._rc_lock     = threading.Lock()
        self._rc_lock_evt = False
        self._rc_rel_evt  = False
        self._rc_filter   = 'any'
        self._prev_ch7    = False
        self._prev_ch8    = False

        self._telem_thread = threading.Thread(target=self._stream_telemetry, daemon=True)
        self._telem_thread.start()
        return True

    def _stream_telemetry(self):
        from pymavlink import mavutil
        while self.running:
            try:
                # Single unified loop processing all required packets.... Added RC_CHANNELS_RAW to fix rc commands sent for lock, release, and filter not been received.
                msg = self.master.recv_match(type=['HEARTBEAT', 'GLOBAL_POSITION_INT', 'SYS_STATUS', 'GPS_RAW_INT', 'RC_CHANNELS', 'RC_CHANNELS_RAW'], blocking=True, timeout=0.1)
                if not msg: continue
                
                msg_type = msg.get_type()
                if msg_type == 'HEARTBEAT':
                    self._telemetry_data["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    self._telemetry_data["mode"] = str(msg.custom_mode)
                
                elif msg_type == 'GLOBAL_POSITION_INT':
                    self._telemetry_data["alt"] = msg.relative_alt / 1000.0  
                    self._telemetry_data["vx"] = msg.vx / 100.0
                    self._telemetry_data["vy"] = msg.vy / 100.0
                    self._telemetry_data["vz"] = msg.vz / 100.0
                
                elif msg_type == 'SYS_STATUS':
                    self._telemetry_data["battery"] = msg.voltage_battery / 1000.0  
                
                elif msg_type == 'GPS_RAW_INT':
                    self._telemetry_data["gps_sats"] = msg.satellites_visible
                # Where we get RC_CHANNELS_RAW from
                elif msg_type in ['RC_CHANNELS', 'RC_CHANNELS_RAW']:
                    with self._rc_lock:
                        # Safely extract channels, defaulting to 0 if missing
                        c7 = getattr(msg, 'chan7_raw', 0)
                        c8 = getattr(msg, 'chan8_raw', 0)
                        c9 = getattr(msg, 'chan9_raw', 0)

                        ch7_high = c7 > 1700
                        ch8_high = c8 > 1700
                        
                        # Edge detection for switches
                        if ch7_high and not self._prev_ch7: self._rc_lock_evt = True
                        if ch8_high and not self._prev_ch8: self._rc_rel_evt = True
                        
                        # Tri-state filter logic
                        if c9 > 0:  # Ensure we actually have a CH9 reading
                            if c9 < 1300:   self._rc_filter = 'any'
                            elif c9 > 1700: self._rc_filter = 'vehicle'
                            else:           self._rc_filter = 'person'

                        self._prev_ch7, self._prev_ch8 = ch7_high, ch8_high

            except Exception: pass

    def get_telemetry(self) -> dict:
        return self._telemetry_data

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
            if self._telemetry_data.get("alt", 0.0) >= (altitude_m - 0.4): break
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


class PX4Backend(FCBackend):
    # PX4 Async implementation remains the same
    def _connect_impl(self) -> bool: return True
    def _execute_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): pass
    def get_telemetry(self) -> dict: return {}
    def consume_rc_events(self) -> tuple: return False, False, 'any'
    def arm_and_takeoff(self, altitude_m: float) -> bool: return True
    def land(self) -> bool: return True


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
    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): self.backend.send_velocity(vx, vy, vz, yaw_rate)
    def land(self) -> bool: return self.backend.land()
    def disconnect(self): self.backend.disconnect()
