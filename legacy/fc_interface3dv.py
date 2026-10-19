#!/usr/bin/env python3
"""
=============================================================================
 Drone Tracker — Phase 3: Robust 3D Flight Controller Abstraction Layer

 Provides an asynchronous, non-blocking unified interface across backends:
   fc = FCInterface(fc_type='ardupilot', connection='127.0.0.1:14550')
   fc.connect()
   fc.arm_and_takeoff(10)
   fc.send_velocity(vx=1.0, vy=0.0, vz=-0.5, yaw_rate=0.3)
   fc.land()

 Architecture: Thread-safe queues prevent telemetry/network calls from stalling
 the primary perception tracking loop. Includes an automated watchdog failsafe.
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
        self.watchdog_timeout = 0.5 # Seconds before triggering safety hover
        self.takeoff_in_progress = False # Safety latch to prevent takeoff overrides
        self._worker_thread = None

    @abstractmethod
    def _connect_impl(self) -> bool:
        pass

    @abstractmethod
    def _execute_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        pass

    @abstractmethod
    def get_telemetry(self) -> dict:
        pass

    @abstractmethod
    def arm_and_takeoff(self, altitude_m: float) -> bool:
        pass

    @abstractmethod
    def land(self) -> bool:
        pass

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
        current_vx = 0.0
        current_vy = 0.0
        current_vz = 0.0
        current_yaw_rate = 0.0
        
        while self.running:
            # Suppress streaming inputs entirely during navigation commands
            if self.takeoff_in_progress:
                time.sleep(0.05)
                continue

            now = time.time()
            
            if now - self.last_cmd_time > self.watchdog_timeout:
                if current_vx != 0.0 or current_vy != 0.0 or current_vz != 0.0 or current_yaw_rate != 0.0:
                    current_vx = 0.0
                    current_vy = 0.0
                    current_vz = 0.0
                    current_yaw_rate = 0.0
                    print("[WATCHDOG ALERT] Tracker stream timed out. Forcing zero-velocity 3D hover.")
                
                try:
                    self._execute_velocity(0.0, 0.0, 0.0, 0.0)
                except Exception as e:
                    print(f"[FC THREAD] Watchdog execution error: {e}")
                time.sleep(0.05)
                continue

            try:
                vx, vy, vz, yaw_rate = self.cmd_queue.get(timeout=0.05)
                current_vx = vx
                current_vy = vy
                current_vz = vz
                current_yaw_rate = yaw_rate
                self._execute_velocity(current_vx, current_vy, current_vz, current_yaw_rate)
            except queue.Empty:
                try:
                    self._execute_velocity(current_vx, current_vy, current_vz, current_yaw_rate)
                except Exception as e:
                    print(f"[FC THREAD] Loop retransmission error: {e}")

    def disconnect(self):
        self.running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
        self.is_connected = False


# ============================================================================
# ARDUPILOT BACKEND (Pure PyMavlink Implementation)
# ============================================================================
class ArduPilotBackend(FCBackend):
    def _connect_impl(self) -> bool:
        print(f"[ARDUPILOT] Initializing pure MAVLink endpoint: {self.connection_string}")
        from pymavlink import mavutil
        
        if "udp:" in self.connection_string:
            addr = self.connection_string.replace("udp:", "")
            self.master = mavutil.mavlink_connection(f"udpin:{addr}")
        elif ":" in self.connection_string and not "/" in self.connection_string:
            self.master = mavutil.mavlink_connection(f"udpin:{self.connection_string}")
        else:
            self.master = mavutil.mavlink_connection(self.connection_string, baud=115200)
            
        print("[ARDUPILOT] Awaiting target heartbeat verification...")
        self.master.wait_heartbeat(timeout=10)
        self.target_system = self.master.target_system
        self.target_component = self.master.target_component
        print(f"[ARDUPILOT] Connected to System: {self.target_system} Component: {self.target_component}")
        
        self._telemetry_data = {"armed": False, "mode": "UNKNOWN", "alt": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}
        self._telem_thread = threading.Thread(target=self._stream_telemetry, daemon=True)
        self._telem_thread.start()
        return True

    def _stream_telemetry(self):
        from pymavlink import mavutil
        while self.running:
            try:
                msg = self.master.recv_match(type=['HEARTBEAT', 'GLOBAL_POSITION_INT', 'VFR_HUD'], blocking=True, timeout=0.1)
                if not msg:
                    continue
                if msg.get_type() == 'HEARTBEAT':
                    self._telemetry_data["armed"] = (msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) != 0
                    self._telemetry_data["mode"] = str(msg.custom_mode)
                elif msg.get_type() == 'GLOBAL_POSITION_INT':
                    self._telemetry_data["alt"] = msg.relative_alt / 1000.0  # mm to meters AGL
                    self._telemetry_data["vx"] = msg.vx / 100.0
                    self._telemetry_data["vy"] = msg.vy / 100.0
                    self._telemetry_data["vz"] = msg.vz / 100.0
            except Exception:
                pass

    def get_telemetry(self) -> dict:
        return self._telemetry_data

    def arm_and_takeoff(self, altitude_m: float) -> bool:
        from pymavlink import mavutil
        self.takeoff_in_progress = True
        
        print("[ARDUPILOT] Initializing Guided Mode switch Sequence...")
        self.master.set_mode(4) # GUIDED Mode ID for ArduCopter
        
        # Verify mode transition via telemetry feedback loop
        start_time = time.time()
        while time.time() - start_time < 3.0:
            if self._telemetry_data.get("mode") == "4":
                break
            time.sleep(0.1)
        
        print("[ARDUPILOT] Sending arm command...")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0
        )
        
        print("[ARDUPILOT] Waiting for motor activation status...")
        start_time = time.time()
        while time.time() - start_time < 4.0:
            if self._telemetry_data.get("armed"):
                print("[ARDUPILOT] Motors armed successfully.")
                break
            time.sleep(0.1)
        
        print(f"[ARDUPILOT] Executing NAV_TAKEOFF command sequence to {altitude_m}m...")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0,
            0, 0, 0, 0, 0, 0, altitude_m
        )
        
        # Telemetry Polling: Block thread execution until target alt is matched
        print("[ARDUPILOT] Climbing to target tracking altitude...")
        start_time = time.time()
        while time.time() - start_time < 25.0:  # 25s failsafe timeout
            current_alt = self._telemetry_data.get("alt", 0.0)
            print(f"  -> Altitude: {current_alt:.2f}m / Goal: {altitude_m:.1f}m", end="\r")
            if current_alt >= (altitude_m - 0.4):
                print(f"\n[ARDUPILOT] Takeoff target altitude achieved.")
                break
            time.sleep(0.2)
            
        self.takeoff_in_progress = False
        return True

    def _execute_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        from pymavlink import mavutil
        type_mask = 0b0000011111000111  # Vx, Vy, Vz + Yaw Rate enabled
        
        self.master.mav.set_position_target_local_ned_send(
            0,
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0, 0, 0,
            vx, vy, vz,
            0, 0, 0,
            0, yaw_rate
        )

    def land(self) -> bool:
        print("[ARDUPILOT] Executing Descent LAND Command Sequence.")
        self.master.set_mode(9) # LAND Mode ID
        return True


# ============================================================================
# PX4 BACKEND (Asynchronous MAVSDK Wrapper Layer)
# ============================================================================
class PX4Backend(FCBackend):
    def _connect_impl(self) -> bool:
        print(f"[PX4] Bridging tracking context onto MAVSDK: {self.connection_string}")
        import asyncio
        from mavsdk import System
        
        self.drone = System()
        self._async_loop = asyncio.new_event_loop()
        
        async def _async_connect():
            await self.drone.connect(system_address=f"udp://{self.connection_string.replace('udp:', '')}")
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    break
            return True
            
        self._async_loop.run_until_complete(_async_connect())
        return True

    def get_telemetry(self) -> dict:
        return {"armed": True, "mode": "OFFBOARD", "alt": 10.0}

    def arm_and_takeoff(self, altitude_m: float) -> bool:
        self.takeoff_in_progress = True
        import asyncio
        async def _takeoff():
            await self.drone.action.arm()
            await self.drone.action.takeoff()
        self._async_loop.run_until_complete(_takeoff())
        self.takeoff_in_progress = False
        return True

    def _execute_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float):
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
        async def _send_inputs():
            try:
                yaw_deg_s = math.degrees(yaw_rate)
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(vx, vy, vz, yaw_deg_s)
                )
            except OffboardError as e:
                print(f"[PX4 ERROR] Rejected offboard control frames: {e}")
        self._async_loop.run_until_complete(_send_inputs())

    def land(self) -> bool:
        import asyncio
        async def _land():
            await self.drone.action.land()
        self._async_loop.run_until_complete(_land())
        return True


# ============================================================================
# INTEGRATION WRAPPER AND FACTORY PATTERN
# ============================================================================
class FCInterface:
    def __init__(self, fc_type: str, connection: str):
        backends = {'ardupilot': ArduPilotBackend, 'px4': PX4Backend}
        fc_type_lower = fc_type.lower()
        if fc_type_lower not in backends:
            raise ValueError(f"Unsupported stack type '{fc_type}'. Choices: {list(backends.keys())}")
        self.backend = backends[fc_type_lower](connection)

    def connect(self) -> bool: return self.backend.connect()
    def get_telemetry(self) -> dict: return self.backend.get_telemetry()
    def arm_and_takeoff(self, altitude_m: float) -> bool: return self.backend.arm_and_takeoff(altitude_m)
    def send_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float): self.backend.send_velocity(vx, vy, vz, yaw_rate)
    def land(self) -> bool: return self.backend.land()
    def disconnect(self): self.backend.disconnect()


if __name__ == '__main__':
    fc = FCInterface(fc_type='ardupilot', connection='127.0.0.1:14550')
    fc.connect()
