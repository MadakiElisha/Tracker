#!/usr/bin/env python3
"""
.....
 Drone Tracker — FC Abstraction Layer

 Provides a single unified interface:
   fc = FCInterface(fc_type='ardupilot', connection='...')
   fc.connect()
   fc.arm_and_takeoff(10)
   fc.send_velocity(vx=1.0, yaw_rate=0.3)
   fc.land()

 Supported backends:
   'ardupilot' — MAVLink via DroneKit (SITL or real hardware)
   'px4'       — MAVLink via MAVSDK (SITL or real hardware)
   'inav'      — MSP via serial (real hardware only)
   'betaflight' — MSP RC injection via serial (real hardware only)
....
"""

import time
import math
import threading
from abc import ABC, abstractmethod

# .....
# BASE CLASS — defines the interface all backends must follow
# .....
class FCBackend(ABC):

    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def arm_and_takeoff(self, altitude_m: float): pass

    @abstractmethod
    def send_velocity(self, vx: float, yaw_rate: float): pass

    @abstractmethod
    def land(self): pass

    @abstractmethod
    def get_telemetry(self) -> dict: pass

    @abstractmethod
    def is_connected(self) -> bool: pass


# .....
# BACKEND 1 — ArduPilot via DroneKit
# .....
class ArduPilotBackend(FCBackend):

    def __init__(self, connection_string='127.0.0.1:14550'):
        self.conn_str = connection_string
        self.vehicle  = None
        self._lock    = threading.Lock()

    def connect(self):
        import collections, collections.abc
        collections.MutableMapping   = collections.abc.MutableMapping
        collections.MutableSequence  = collections.abc.MutableSequence
        collections.Callable         = collections.abc.Callable

        from dronekit import connect, VehicleMode
        self._VehicleMode = VehicleMode

        print(f"[ArduPilot] Connecting to {self.conn_str}...")
        self.vehicle = connect(self.conn_str,
                               wait_ready=True,
                               timeout=60)
        print(f"[ArduPilot] Connected. Mode: {self.vehicle.mode.name}")

    def arm_and_takeoff(self, altitude_m: float):
        v = self.vehicle
        VM = self._VehicleMode

        print("[ArduPilot] Setting GUIDED mode...")
        v.mode = VM("GUIDED")
        timeout = time.time() + 10
        while v.mode.name != "GUIDED":
            if time.time() > timeout:
                raise TimeoutError("Could not enter GUIDED mode")
            time.sleep(0.3)

        print("[ArduPilot] Waiting for armable state...")
        timeout = time.time() + 30
        while not v.is_armable:
            if time.time() > timeout:
                raise TimeoutError("Vehicle not armable — check EKF/GPS")
            time.sleep(1)

        print("[ArduPilot] Arming...")
        v.armed = True
        timeout = time.time() + 15
        while not v.armed:
            if time.time() > timeout:
                raise TimeoutError("Arm failed")
            time.sleep(0.5)

        print(f"[ArduPilot] Taking off to {altitude_m}m...")
        v.simple_takeoff(altitude_m)

        while True:
            alt = v.location.global_relative_frame.alt
            print(f"  Altitude: {alt:.1f}m", end='\r')
            if alt >= altitude_m * 0.95:
                break
            time.sleep(0.3)
        print(f"\n[ArduPilot] Reached {altitude_m}m.")

    def send_velocity(self, vx: float, yaw_rate: float):
        """
        Send body-frame velocity command.
        vx        : forward speed m/s (negative = backward)
        yaw_rate  : rotation speed rad/s (positive = clockwise)
        """
        from pymavlink import mavutil
        with self._lock:
            msg = self.vehicle.message_factory\
                      .set_position_target_local_ned_encode(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000011111000111,   # use vx, vy, vz + yaw_rate
                0, 0, 0,
                vx, 0, 0,
                0, 0, 0,
                0, yaw_rate
            )
            self.vehicle.send_mavlink(msg)

    def land(self):
        print("[ArduPilot] Returning to launch...")
        self.vehicle.mode = self._VehicleMode("RTL")

    def get_telemetry(self) -> dict:
        if not self.vehicle:
            return {}
        v = self.vehicle
        return {
            'alt':      round(v.location.global_relative_frame.alt, 2),
            'heading':  v.heading,
            'mode':     v.mode.name,
            'armed':    v.armed,
            'battery':  v.battery.voltage,
            'gps_fix':  v.gps_0.fix_type,
            'gps_sats': v.gps_0.satellites_visible,
            'vx':       v.velocity[0],
            'vy':       v.velocity[1],
            'vz':       v.velocity[2],
        }

    def is_connected(self) -> bool:
        return self.vehicle is not None

    def close(self):
        if self.vehicle:
            self.vehicle.close()


# .....
# BACKEND 2 — PX4 via MAVSDK-Python
# (Same MAVLink protocol, different abstraction library)
# .....
class PX4Backend(FCBackend):
    """
    PX4 backend using MAVSDK-Python.
    Install: pip install mavsdk
    Connection: 'udp://:14540' for SITL
    """

    def __init__(self, connection_string='udp://:14540'):
        self.conn_str = connection_string
        self._drone   = None
        self._loop    = None

    def connect(self):
        import asyncio
        from mavsdk import System

        self._loop  = asyncio.new_event_loop()
        self._drone = System()

        async def _connect():
            await self._drone.connect(
                system_address=self.conn_str)
            print("[PX4] Waiting for connection...")
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    print("[PX4] Connected.")
                    break

        self._loop.run_until_complete(_connect())

    def arm_and_takeoff(self, altitude_m: float):
        import asyncio

        async def _fly():
            await self._drone.action.arm()
            await self._drone.action.takeoff()
            await asyncio.sleep(altitude_m / 2)  # rough wait

        self._loop.run_until_complete(_fly())

    def send_velocity(self, vx: float, yaw_rate: float):
        import asyncio
        from mavsdk.offboard import VelocityBodyYawspeed

        async def _send():
            await self._drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    vx,               # forward m/s
                    0.0,              # right m/s (no strafing)
                    0.0,              # down m/s
                    math.degrees(yaw_rate)  # MAVSDK uses deg/s
                )
            )

        self._loop.run_until_complete(_send())

    def land(self):
        import asyncio
        async def _land():
            await self._drone.action.land()
        self._loop.run_until_complete(_land())

    def get_telemetry(self) -> dict:
        # Simplified — expand as needed
        return {'mode': 'offboard', 'armed': True}

    def is_connected(self) -> bool:
        return self._drone is not None

    def close(self):
        pass


# .....
# BACKEND 3 — iNav via MSP
# .....
class INavBackend(FCBackend):
    """
    iNav backend using MSP protocol over serial.
    iNav has no velocity mode — we inject RC channel values.
    RC mapping (standard iNav):
      CH1: Roll, CH2: Pitch, CH3: Throttle, CH4: Yaw
    """

    MSP_SET_RAW_RC = 200
    RC_MID         = 1500
    RC_MIN         = 1000
    RC_MAX         = 2000

    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        self.port  = port
        self.baud  = baud
        self._ser  = None
        self._lock = threading.Lock()

        # Current RC values — start centred
        self.rc = [1500, 1500, 1000, 1500,   # Roll Pitch Throttle Yaw
                   1000, 1000, 1000, 1000]   # AUX1-4

    def connect(self):
        import serial
        self._ser = serial.Serial(
            self.port, self.baud, timeout=1)
        print(f"[iNav] Connected on {self.port} @ {self.baud}")

    def _send_msp(self, cmd: int, data: bytes = b''):
        """Build and send an MSP packet."""
        size   = len(data)
        chksum = size ^ cmd
        for b in data:
            chksum ^= b

        packet = (b'$M<' +
                  bytes([size, cmd]) +
                  data +
                  bytes([chksum]))
        with self._lock:
            self._ser.write(packet)

    def _set_rc(self):
        """Send current RC values via MSP SET_RAW_RC."""
        import struct
        data = struct.pack('<' + 'H' * 8, *self.rc)
        self._send_msp(self.MSP_SET_RAW_RC, data)

    def arm_and_takeoff(self, altitude_m: float):
        """
        iNav: arm via RC (throttle low + yaw right),
        then increase throttle to hover.
        For SITL testing this is approximate.
        """
        print("[iNav] Arming via RC (throttle low, yaw right)...")
        self.rc[2] = 1000  # throttle low
        self.rc[3] = 2000  # yaw right = arm
        for _ in range(50):
            self._set_rc()
            time.sleep(0.02)

        self.rc[3] = 1500  # yaw centre
        time.sleep(0.5)

        print("[iNav] Increasing throttle to hover...")
        for throttle in range(1000, 1550, 10):
            self.rc[2] = throttle
            self._set_rc()
            time.sleep(0.05)

        print("[iNav] Hovering. No barometer feedback in MSP mode.")

    def send_velocity(self, vx: float, yaw_rate: float):
        """
        Map velocity commands to RC values.
        vx        → Pitch channel (forward/back)
        yaw_rate  → Yaw channel
        """
        # Scale: 1.0 m/s → 100us RC deflection
        pitch_deflect = int(vx * 100)
        yaw_deflect   = int(math.degrees(yaw_rate) * 3)

        self.rc[1] = max(self.RC_MIN,
                         min(self.RC_MAX,
                             self.RC_MID + pitch_deflect))
        self.rc[3] = max(self.RC_MIN,
                         min(self.RC_MAX,
                             self.RC_MID + yaw_deflect))
        self._set_rc()

    def land(self):
        print("[iNav] Descending...")
        for throttle in range(self.rc[2], 1000, -10):
            self.rc[2] = throttle
            self._set_rc()
            time.sleep(0.05)

    def get_telemetry(self) -> dict:
        return {
            'rc_pitch':    self.rc[1],
            'rc_throttle': self.rc[2],
            'rc_yaw':      self.rc[3],
        }

    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def close(self):
        if self._ser:
            self._ser.close()


# .....
# BACKEND 4 — Betaflight via MSP RC injection
# (same MSP protocol as iNav, different tuning)
# .....
class BetaflightBackend(INavBackend):
    """
    Betaflight uses identical MSP protocol to iNav.
    Override port/baud defaults only.
    Betaflight has no autonomous flight modes —
    purely RC injection via MSP.
    """

    def __init__(self, port='/dev/ttyUSB0', baud=115200):
        super().__init__(port, baud)
        print("[Betaflight] Using MSP RC injection mode.")

    def arm_and_takeoff(self, altitude_m: float):
        print("[Betaflight] WARNING: No altitude hold in Betaflight.")
        print("  Ensure Angle/Horizon mode is set on AUX channel.")
        super().arm_and_takeoff(altitude_m)


# .....
# FACTORY — instantiate correct backend from string
# .....
def FCInterface(fc_type: str, connection: str) -> FCBackend:
    """
    Factory function. Returns the correct backend.

    Usage:
        fc = FCInterface('ardupilot', '127.0.0.1:14550')
        fc = FCInterface('px4',       'udp://:14540')
        fc = FCInterface('inav',      '/dev/ttyUSB0')
        fc = FCInterface('betaflight','/dev/ttyUSB0')
    """
    fc_type = fc_type.lower().strip()
    backends = {
        'ardupilot':   ArduPilotBackend,
        'px4':         PX4Backend,
        'inav':        INavBackend,
        'betaflight':  BetaflightBackend,
    }
    if fc_type not in backends:
        raise ValueError(
            f"Unknown FC type '{fc_type}'. "
            f"Choose from: {list(backends.keys())}")
    return backends[fc_type](connection)


# .....
# QUICK TEST — run directly to verify connection
# .....
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc',
                        default='ardupilot',
                        help='FC type: ardupilot|px4|inav|betaflight')
    parser.add_argument('--connect',
                        default='127.0.0.1:14550',
                        help='Connection string')
    parser.add_argument('--takeoff',
                        action='store_true',
                        help='Arm and takeoff to 10m then RTL')
    args = parser.parse_args()

    fc = FCInterface(args.fc, args.connect)
    fc.connect()

    telem = fc.get_telemetry()
    print("\n[TELEMETRY]")
    for k, v in telem.items():
        print(f"  {k:12s}: {v}")

    if args.takeoff:
        fc.arm_and_takeoff(10)
        print("\nHovering for 5 seconds...")
        for i in range(5):
            telem = fc.get_telemetry()
            print(f"  t+{i+1}s  alt={telem.get('alt', '?')}m  "
                  f"mode={telem.get('mode', '?')}")
            time.sleep(1)
        fc.land()

    print("\n[FC INTERFACE] Test complete.")
