#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Unified Flight Controller Interface (HAL)
 
 Combines Factory Design Pattern with high-frequency, 
 non-blocking background command queues and safety watchdogs.
=============================================================
"""

import time
import math
import threading
from abc import ABC, abstractmethod
from pymavlink import mavutil

# ===========================================================
# BASE INTERFACE CONTRACT
# ===========================================================
class FCBackend(ABC):
    @abstractmethod
    def connect(self) -> bool: pass

    @abstractmethod
    def arm_and_takeoff(self, altitude_m: float): pass

    @abstractmethod
    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): pass

    @abstractmethod
    def land(self): pass

    @abstractmethod
    def get_telemetry(self) -> dict: pass

    @abstractmethod
    def close(self): pass


# ===========================================================
# PRODUCTION-READY ARDUPILOT BACKEND (Native Pymavlink)
# ===========================================================
class ArduPilotBackend(FCBackend):
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.vehicle = None
        self.is_running = False
        
        # Thread-safe target states
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vz = 0.0
        self._target_yaw_rate = 0.0
        self._last_command_time = 0.0
        self._lock = threading.Lock()
        
        # Thread-safe telemetry state
        self._telemetry = {"armed": False, "alt": 0.0, "mode": "UNKNOWN"}
        
        # Watchdog parameters
        self.watchdog_timeout = 1.0  # Safe hover if stream drops for 1s
        self._tx_thread = None
        self._rx_thread = None

    def connect(self) -> bool:
        print(f"[AP_BACKEND] Connecting via MAVLink to {self.conn_string}...")
        try:
            self.vehicle = mavutil.mavlink_connection(self.conn_string)
            self.vehicle.wait_heartbeat(timeout=10)
            print(f"[AP_BACKEND] Link verified. Target System ID: {self.vehicle.target_system}")
            
            self.is_running = True
            
            # Thread 1: Transmits Offboard setpoints to FC at 10Hz
            self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
            self._tx_thread.start()
            
            # Thread 2: Asynchronously parses incoming telemetry data
            self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
            self._rx_thread.start()
            
            return True
        except Exception as e:
            print(f"[AP_BACKEND] Connection failed: {e}")
            return False

    def _tx_loop(self):
        """ Runs constantly in the background. Frees the main pipeline thread. """
        while self.is_running:
            now = time.time()
            with self._lock:
                # Watchdog Intervention: force zero-velocity hover if perception drops out
                if now - self._last_command_time > self.watchdog_timeout:
                    vx, vy, vz, yr = 0.0, 0.0, 0.0, 0.0
                else:
                    vx, vy, vz, yr = self._target_vx, self._target_vy, self._target_vz, self._target_yaw_rate

            if self.vehicle:
                # Coordinate Frame: MAV_FRAME_BODY_NED (Relative to the drone's heading nose)
                self.vehicle.mav.set_position_target_local_ned_send(
                    0, 
                    self.vehicle.target_system, 
                    self.vehicle.target_component,
                    mavutil.mavlink.MAV_FRAME_BODY_NED,
                    0b0000011111000111,  # Ignore Position/Acc, track Vel + Yaw Rate
                    0, 0, 0,             # Local X, Y, Z positions (unused)
                    vx, vy, vz,          # m/s (Forward, Right, Down)
                    0, 0, 0,             # Accelerations (unused)
                    0, yr                # Yaw, Yaw Rate (rad/s)
                )
            time.sleep(0.1)  # 10Hz command streaming frequency

    def _rx_loop(self):
        """ Updates internal telemetry state instantly without locking the caller. """
        while self.is_running:
            msg = self.vehicle.recv_match(type=['GLOBAL_POSITION_INT', 'HEARTBEAT'], blocking=True, timeout=0.1)
            if not msg:
                continue
                
            with self._lock:
                if msg.get_type() == 'GLOBAL_POSITION_INT':
                    self._telemetry["alt"] = msg.relative_alt / 1000.0  # Convert mm to meters
                elif msg.get_type() == 'HEARTBEAT':
                    self._telemetry["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    # Basic mode mapping resolution
                    self._telemetry["mode"] = self.vehicle.flightmode

    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        """ Non-blocking. Drops updates into memory quickly for the background loop to catch. """
        with self._lock:
            self._target_vx = vx
            self._target_vy = vy
            self._target_vz = vz
            self._target_yaw_rate = yaw_rate
            self._last_command_time = time.time()

    def arm_and_takeoff(self, altitude_m: float):
        print(f"[AP_BACKEND] Arming and initiating takeoff to {altitude_m}m...")
        # Change mode to GUIDED
        mode_id = self.vehicle.mode_mapping()['GUIDED']
        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        
        # Arm Command
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        
        # Takeoff Command
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude_m)

    def land(self):
        print("[AP_BACKEND] Executing LAND sequence.")
        mode_id = self.vehicle.mode_mapping()['LAND']
        self.vehicle.mav.set_mode_send(self.vehicle.target_system, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)

    def get_telemetry(self) -> dict:
        with self._lock:
            return self._telemetry.copy()

    def close(self):
        self.is_running = False
        if self._tx_thread: self._tx_thread.join()
        if self._rx_thread: self._rx_thread.join()
        print("[AP_BACKEND] Interface successfully down.")


# ===========================================================
# STUB ARCHITECTURE FOR FUTURE BACKENDS
# ===========================================================
class PX4Backend(FCBackend):
    def __init__(self, connection_string: str): self.conn_string = connection_string
    def connect(self) -> bool: print("[PX4] Multi-thread layer bound."); return True
    def arm_and_takeoff(self, altitude_m: float): pass
    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): pass
    def land(self): pass
    def get_telemetry(self) -> dict: return {"armed": False, "alt": 0.0, "mode": "OFFBOARD"}
    def close(self): pass

class INavBackend(FCBackend):
    def __init__(self, connection_string: str): self.conn_string = connection_string
    def connect(self) -> bool: return True
    def arm_and_takeoff(self, altitude_m: float): pass
    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): pass
    def land(self): pass
    def get_telemetry(self) -> dict: return {}
    def close(self): pass

class BetaflightBackend(FCBackend):
    def __init__(self, connection_string: str): self.conn_string = connection_string
    def connect(self) -> bool: return True
    def arm_and_takeoff(self, altitude_m: float): pass
    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): pass
    def land(self): pass
    def get_telemetry(self) -> dict: return {}
    def close(self): pass


# ===========================================================
# THE FACTORY CONTROLLER
# ===========================================================
def FCInterface(fc_type: str, connection_string: str) -> FCBackend:
    backends = {
        "ardupilot":  ArduPilotBackend,
        "px4":        PX4Backend,
        "inav":       INavBackend,
        "betaflight": BetaflightBackend
    }
    
    clean_type = fc_type.lower().strip()
    if clean_type not in backends:
        raise ValueError(f"Target FC architecture '{fc_type}' is unsupported.")
        
    return backends[clean_type](connection_string)


# ===========================================================
# IN-LINE STANDALONE VERIFICATION
# ===========================================================
if __name__ == '__main__':
    # Initialize factory with the desired target profile
    fc = FCInterface(fc_type='ardupilot', connection_string='127.0.0.1:14550')
    
    if fc.connect():
        # Spin loop parsing background execution behavior
        for _ in range(5):
            print(f"[TEST RUN] Local Telemetry: {fc.get_telemetry()}")
            time.sleep(0.5)
        fc.close()
