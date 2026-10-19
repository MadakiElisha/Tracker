#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Phase 3: High-Performance FC Interface
 
 Features:
   - Threaded Command Loop: Never blocks the Vision Pipeline.
   - Watchdog/Failsafe: Auto-brakes if perception crashes.
   - Mavlink-Native: Uses Pymavlink directly for speed/control.
=============================================================
"""

import time
import math
import threading
from pymavlink import mavutil

class FCInterface:
    def __init__(self, connection_string='127.0.0.1:14550', heartbeat_timeout=1.0):
        self.conn_string = connection_string
        self.vehicle = None
        self.is_running = False
        
        # Target State (Thread-Safe)
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vz = 0.0
        self._target_yaw_rate = 0.0
        self._last_command_time = 0.0
        self._lock = threading.Lock()
        
        # Failsafe Settings
        self.watchdog_threshold = heartbeat_timeout # Seconds before auto-hover
        
    def connect(self):
        print(f"[FC] Connecting to {self.conn_string}...")
        # Supports TCP, UDP, or Serial (/dev/ttyUSB0)
        self.vehicle = mavutil.mavlink_connection(self.conn_string)
        self.vehicle.wait_heartbeat()
        print(f"[FC] Heartbeat received! System {self.vehicle.target_system}")
        
        # Start the background command thread
        self.is_running = True
        self.thread = threading.Thread(target=self._control_loop, daemon=True)
        self.thread.start()

    def _control_loop(self):
        """
        Background thread that streams MAVLink packets to the FC at a constant 10Hz.
        This prevents the 'Command Lag' seen in DroneKit.
        """
        print("[FC] Background control loop started.")
        while self.is_running:
            now = time.time()
            
            with self._lock:
                # WATCHDOG: If perception hasn't sent a command recently, force a hover.
                if now - self._last_command_time > self.watchdog_threshold:
                    vx, vy, vz, yr = 0.0, 0.0, 0.0, 0.0
                else:
                    vx, vy, vz, yr = self._target_vx, self._target_vy, self._target_vz, self._target_yaw_rate

            # Send SET_POSITION_TARGET_LOCAL_NED
            # Coordinate Frame: MAV_FRAME_BODY_NED (Forward, Right, Down relative to drone)
            self.vehicle.mav.set_position_target_local_ned_send(
                0,                          # time_boot_ms
                self.vehicle.target_system, 
                self.vehicle.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000011111000111,         # Bitmask: Use only Vx, Vy, Vz, and YawRate
                0, 0, 0,                    # x, y, z positions (ignored)
                vx, vy, vz,                 # velocities (m/s)
                0, 0, 0,                    # accelerations (ignored)
                0, yr                       # yaw, yaw_rate (rad/s)
            )
            
            time.sleep(0.1) # Constant 10Hz stream is standard for MAVLink offboard

    def send_velocity(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        """
        Non-blocking: Just updates the internal target. 
        Safe to call from within your YOLO loop.
        """
        with self._lock:
            self._target_vx = vx
            self._target_vy = vy
            self._target_vz = vz
            self._target_yaw_rate = yaw_rate
            self._last_command_time = time.time()

    def arm_and_takeoff(self, altitude):
        print(f"[FC] Arming and Taking off to {altitude}m...")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
        
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, 0, altitude)

    def set_mode(self, mode_name):
        """ GUIDED for ArduPilot or OFFBOARD for PX4 """
        mode_id = self.vehicle.mode_mapping()[mode_name]
        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)

    def close(self):
        self.is_running = False
        if self.thread.is_alive():
            self.thread.join()
        print("[FC] Interface closed.")

# ============================================================
# USAGE EXAMPLE (How to integrate with your Phase 2 script)
# ============================================================
if __name__ == "__main__":
    # 1. Initialize
    fc = FCInterface(connection_string='127.0.0.1:14550')
    fc.connect()
    
    # 2. Preparation
    fc.set_mode('GUIDED')
    fc.arm_and_takeoff(5)
    time.sleep(5)
    
    # 3. Simulate a Perception Loop
    print("[TEST] Sending forward velocity for 3 seconds...")
    start = time.time()
    while time.time() - start < 3:
        # This update is instant and doesn't block
        fc.send_velocity(vx=1.5, yaw_rate=0.2) 
        time.sleep(0.03) # Simulating 30fps YOLO
        
    print("[TEST] Simulating a script freeze (Failsafe should kick in)...")
    time.sleep(2) # fc._control_loop will auto-hover because we stopped calling send_velocity
    
    fc.set_mode('RTL')
