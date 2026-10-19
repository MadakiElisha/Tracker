#!/usr/bin/env python3
"""
=============================================================================
 Drone Tracker — Flight Controller Abstraction Layer (hardened)

 Unified interface over flight-controller backends:

   fc = FCInterface('ardupilot', '127.0.0.1:14550', fc_cfg, rc_cfg, safety_cfg)
   fc.connect()
   fc.arm_and_takeoff(10)        # -> bool, verified (mode, armed, altitude)
   fc.send_velocity(vx, 0, vz, yaw_rate)
   fc.hover(); fc.rtl(); fc.land()
   fc.telemetry_age()            # seconds since last autopilot message

 Threading model
   * telemetry thread  — the ONLY reader of the MAVLink socket
   * command worker    — streams the latest velocity command; zero-velocity
                         watchdog if the tracker stops sending
   * all socket writes go through one lock (_tx_lock)

 Hardening vs the previous version
   * logging + ExceptionCounter instead of print()/silent `except: pass`
   * takeoff verifies GUIDED mode, arming and altitude, has timeouts and an
     abort event, and returns False on failure
   * RC edge detection uses a calibration window (no spurious startup events),
     rejects invalid PWM values, thresholds come from config
   * RTL is a real RTL (mode 6); land() is LAND (mode 9)
   * link-age tracking for the safety supervisor
   * PX4 backend is an explicit NotImplementedError, not a stub that
     pretends to arm
=============================================================================
"""

import queue
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

from tracker_logger import ExceptionCounter, get_logger

log = get_logger("fc")

ARDUCOPTER_MODE_GUIDED = 4
ARDUCOPTER_MODE_RTL    = 6
ARDUCOPTER_MODE_LAND   = 9


# ============================================================================
# RC DECODER (pure logic — unit-testable without MAVLink)
# ============================================================================
class RCDecoder:
    """
    Turns raw RC channel PWM values into lock/release edge events and a
    3-position class filter.

      * the first `cal_frames` frames only record switch state, so a switch
        that is already HIGH at start-up (SITL CH8 defaults to 1800) can't fire
      * PWM outside 800..2200 (0 / 65535 = no RC) is ignored, not treated as LOW
      * events are latched until consume() is called
    """
    def __init__(self, rc_cfg: Optional[dict] = None, cal_frames: int = 10):
        c = rc_cfg or {}
        self.hi, self.lo = c.get('high_threshold', 1700), c.get('low_threshold', 1300)
        self.ch_lock    = c.get('lock_channel', 7)
        self.ch_release = c.get('release_channel', 8)
        self.ch_filter  = c.get('filter_channel', 9)
        self.cal_frames = cal_frames
        self._lock = threading.Lock()
        self._frames = 0
        self._prev_hi = {}
        self._lock_evt = self._rel_evt = False
        self._filter = 'any'

    @staticmethod
    def _valid(v):
        return v if v is not None and 800 <= v <= 2200 else None

    def feed(self, channels: dict):
        """channels: {channel_number: pwm}"""
        with self._lock:
            self._frames += 1
            calibrating = self._frames <= self.cal_frames
            for ch in (self.ch_lock, self.ch_release):
                v = self._valid(channels.get(ch))
                if v is None:
                    continue
                is_hi = v > self.hi
                if is_hi and not self._prev_hi.get(ch, False) and not calibrating:
                    if ch == self.ch_lock: self._lock_evt = True
                    else:                  self._rel_evt  = True
                self._prev_hi[ch] = is_hi
            v = self._valid(channels.get(self.ch_filter))
            if v is not None:
                self._filter = 'any' if v < self.lo else ('vehicle' if v > self.hi else 'person')

    def consume(self) -> tuple:
        with self._lock:
            out = (self._lock_evt, self._rel_evt, self._filter)
            self._lock_evt = self._rel_evt = False
            return out


# ============================================================================
# BASE BACKEND
# ============================================================================
class FCBackend(ABC):
    def __init__(self, connection_string: str,
                 fc_cfg: Optional[dict] = None,
                 rc_cfg: Optional[dict] = None,
                 safety_cfg: Optional[dict] = None):
        self.connection_string = connection_string
        self.fc_cfg     = fc_cfg or {}
        self.rc_cfg     = rc_cfg or {}
        self.safety_cfg = safety_cfg or {}

        self.cmd_queue = queue.Queue(maxsize=1)
        self.is_connected = False
        self.running = False
        self.last_cmd_time = 0.0
        self.watchdog_timeout = float(self.fc_cfg.get('watchdog_timeout', 0.5))
        self.takeoff_in_progress = False
        self.abort_takeoff = threading.Event()

        self._worker_thread = None
        self._telem_thread  = None
        self.errors = ExceptionCounter()

    # -- backend contract ---------------------------------------------------
    @abstractmethod
    def _connect_impl(self) -> bool: ...
    @abstractmethod
    def _stream_telemetry(self): ...
    @abstractmethod
    def _execute_velocity(self, vx, vy, vz, yaw_rate): ...
    @abstractmethod
    def get_telemetry(self) -> dict: ...
    @abstractmethod
    def telemetry_age(self) -> Optional[float]: ...
    @abstractmethod
    def consume_rc_events(self) -> tuple: ...
    @abstractmethod
    def arm_and_takeoff(self, altitude_m: float) -> bool: ...
    @abstractmethod
    def rtl(self) -> bool: ...
    @abstractmethod
    def land(self) -> bool: ...

    # -- lifecycle ----------------------------------------------------------
    def connect(self) -> bool:
        if not self._connect_impl():
            return False
        self.is_connected = True
        self.running = True            # MUST precede thread start
        self.last_cmd_time = time.monotonic()
        self._telem_thread = threading.Thread(
            target=self._stream_telemetry, name="fc-telem", daemon=True)
        self._telem_thread.start()
        self._worker_thread = threading.Thread(
            target=self._backend_worker, name="fc-cmd", daemon=True)
        self._worker_thread.start()
        log.info("Telemetry and command threads started.")
        return True

    def disconnect(self):
        self.running = False
        for t in (self._telem_thread, self._worker_thread):
            if t:
                t.join(timeout=1.0)
        self.is_connected = False
        log.info("Disconnected. FC-layer errors: %s", self.errors.summary())

    # -- commands -----------------------------------------------------------
    def send_velocity(self, vx, vy, vz, yaw_rate):
        if not self.is_connected or self.takeoff_in_progress:
            return
        self._put_latest((vx, vy, vz, yaw_rate))
        self.last_cmd_time = time.monotonic()

    def hover(self):
        """Immediately command zero velocity and keep streaming it."""
        self._put_latest((0.0, 0.0, 0.0, 0.0))
        self.last_cmd_time = time.monotonic()

    def _put_latest(self, cmd):
        try:
            while True:
                self.cmd_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.cmd_queue.put_nowait(cmd)
        except queue.Full:
            pass   # another producer won the race; its command is newer

    def _backend_worker(self):
        cur = (0.0, 0.0, 0.0, 0.0)
        was_stale = False
        while self.running:
            if self.takeoff_in_progress:
                time.sleep(0.05)
                continue
            armed = self.get_telemetry().get("armed", False)

            if time.monotonic() - self.last_cmd_time > self.watchdog_timeout:
                if any(v != 0.0 for v in cur):
                    cur = (0.0, 0.0, 0.0, 0.0)
                if not was_stale:
                    log.warning("Command stream stale (>%.2fs): holding zero velocity.",
                                self.watchdog_timeout)
                    was_stale = True
                if armed:
                    self._safe_send(cur)
                time.sleep(0.05)
                continue
            was_stale = False

            try:
                cur = self.cmd_queue.get(timeout=0.05)
            except queue.Empty:
                pass                       # re-send previous command below
            if armed:
                self._safe_send(cur)

    def _safe_send(self, cmd):
        try:
            self._execute_velocity(*cmd)
        except Exception as e:
            self.errors.record('velocity_tx', e)


# ============================================================================
# ARDUPILOT (pymavlink)
# ============================================================================
class ArduPilotBackend(FCBackend):

    def _connect_impl(self) -> bool:
        from pymavlink import mavutil
        self._mavutil = mavutil
        cs = self.connection_string
        log.info("Opening MAVLink endpoint: %s", cs)

        if "udp:" in cs:
            self.master = mavutil.mavlink_connection(f"udpin:{cs.replace('udp:', '')}")
        elif ":" in cs and "/" not in cs:
            self.master = mavutil.mavlink_connection(f"udpin:{cs}")
        else:
            self.master = mavutil.mavlink_connection(
                cs, baud=int(self.fc_cfg.get('baud', 921600)))

        self._tx_lock = threading.Lock()

        # Wait for a heartbeat whose SOURCE component is the autopilot (1).
        # A bare wait_heartbeat() can lock onto another component and silently
        # break every component-targeted command (this broke RC streaming).
        AP = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
        wait_s = float(self.fc_cfg.get('connect_timeout_s', 15.0))
        deadline = time.monotonic() + wait_s
        found = False
        while time.monotonic() < deadline:
            msg = self.master.recv_match(type='HEARTBEAT', blocking=True, timeout=2.0)
            if msg is None:
                continue
            sysid, comp = msg.get_srcSystem(), msg.get_srcComponent()
            log.debug("Heartbeat seen: system=%d component=%d", sysid, comp)
            if comp == AP:
                self.target_system, self.target_component = sysid, comp
                found = True
                break
        if not found:
            log.error("No autopilot (component 1) heartbeat within %.0f s.", wait_s)
            return False
        log.info("Connected: system=%d component=%d",
                 self.target_system, self.target_component)

        self._send_command(mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                           65, int(1e6 / max(1, self.fc_cfg.get('telemetry_rate_hz', 10))))
        ack = self.master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3.0)
        if ack is None:
            log.warning("No COMMAND_ACK for RC_CHANNELS interval request; RC stream may be inactive.")
        else:
            name = mavutil.mavlink.enums['MAV_RESULT'][ack.result].name
            log.info("SET_MESSAGE_INTERVAL(RC_CHANNELS) -> %s", name)

        self._telem = {"armed": False, "mode": "UNKNOWN", "alt": 0.0,
                       "heading": 0, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                       "battery": 0.0, "gps_sats": 0}
        self._telem_lock = threading.Lock()
        self._last_rx = time.monotonic()

        self._rc = RCDecoder(self.rc_cfg, int(self.fc_cfg.get('rc_cal_frames', 10)))
        self._rc_seen = False
        return True

    # -- send helpers -------------------------------------------------------
    def _send_command(self, cmd, p1=0, p2=0, p3=0, p4=0, p5=0, p6=0, p7=0):
        with self._tx_lock:
            self.master.mav.command_long_send(
                self.target_system, self.target_component, cmd, 0,
                p1, p2, p3, p4, p5, p6, p7)

    def _set_mode(self, mode_id: int):
        with self._tx_lock:
            self.master.set_mode(mode_id)

    def _wait_for(self, predicate, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.abort_takeoff.is_set():
                return False
            if predicate(self.get_telemetry()):
                return True
            time.sleep(0.1)
        return False

    # -- telemetry ----------------------------------------------------------
    def _stream_telemetry(self):
        mavutil = self._mavutil
        silent_warned = False
        while self.running:
            try:
                msg = self.master.recv_match(
                    type=['HEARTBEAT', 'GLOBAL_POSITION_INT', 'SYS_STATUS',
                          'GPS_RAW_INT', 'RC_CHANNELS'],
                    blocking=True, timeout=0.1)
                if msg is None:
                    if not silent_warned and time.monotonic() - self._last_rx > 3.0:
                        log.warning("No MAVLink traffic for 3 s. On WSL2 this is usually a "
                                    "leftover process holding the UDP port (ss -ulnp | grep <port>).")
                        silent_warned = True
                    continue

                t = msg.get_type()
                if t == 'HEARTBEAT' and msg.get_srcComponent() != self.target_component:
                    continue                    # ignore GCS/gimbal/etc. heartbeats
                self._last_rx = time.monotonic()
                silent_warned = False

                if t == 'HEARTBEAT':
                    with self._telem_lock:
                        self._telem["armed"] = bool(
                            msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        self._telem["mode"] = str(msg.custom_mode)
                elif t == 'GLOBAL_POSITION_INT':
                    with self._telem_lock:
                        self._telem["alt"] = msg.relative_alt / 1000.0
                        self._telem["vx"], self._telem["vy"], self._telem["vz"] = (
                            msg.vx / 100.0, msg.vy / 100.0, msg.vz / 100.0)
                        if msg.hdg != 65535:
                            self._telem["heading"] = int(msg.hdg / 100)
                elif t == 'SYS_STATUS':
                    with self._telem_lock:
                        # 65535 / -1 mean "not reported"
                        v = msg.voltage_battery
                        self._telem["battery"] = 0.0 if v in (65535, -1) else v / 1000.0
                elif t == 'GPS_RAW_INT':
                    with self._telem_lock:
                        self._telem["gps_sats"] = msg.satellites_visible
                elif t == 'RC_CHANNELS':
                    self._handle_rc(msg)
            except Exception as e:
                self.errors.record('telemetry_rx', e)
                time.sleep(0.01)

    def _handle_rc(self, msg):
        if not self._rc_seen:
            self._rc_seen = True
            log.info("First RC_CHANNELS message received — RC stream active.")
        self._rc.feed({ch: getattr(msg, f'chan{ch}_raw', None) for ch in range(1, 19)})

    def get_telemetry(self) -> dict:
        with self._telem_lock:
            return dict(self._telem)

    def telemetry_age(self) -> Optional[float]:
        return time.monotonic() - self._last_rx

    def consume_rc_events(self) -> tuple:
        return self._rc.consume()

    # -- flight commands ----------------------------------------------------
    def arm_and_takeoff(self, altitude_m: float) -> bool:
        mavutil = self._mavutil
        timeout = float(self.safety_cfg.get('takeoff_timeout_s', 30.0))
        self.abort_takeoff.clear()
        self.takeoff_in_progress = True
        try:
            log.info("Takeoff: switching to GUIDED...")
            self._set_mode(ARDUCOPTER_MODE_GUIDED)
            if not self._wait_for(lambda t: t["mode"] == str(ARDUCOPTER_MODE_GUIDED),
                                  float(self.safety_cfg.get('mode_timeout_s', 5.0))):
                log.error("Takeoff aborted: GUIDED mode not confirmed.")
                return False

            log.info("Takeoff: arming...")
            self._send_command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            if not self._wait_for(lambda t: t["armed"],
                                  float(self.safety_cfg.get('arm_timeout_s', 15.0))):
                log.error("Takeoff aborted: vehicle did not arm (pre-arm checks / EKF?).")
                return False

            log.info("Takeoff: climbing to %.1f m...", altitude_m)
            self._send_command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, altitude_m)
            if not self._wait_for(lambda t: t["alt"] >= altitude_m - 0.5, timeout):
                log.error("Takeoff aborted: altitude %.1f m not reached in %.0f s.",
                          altitude_m, timeout)
                return False
            log.info("Takeoff complete (alt=%.1f m).", self.get_telemetry()["alt"])
            return True
        except Exception as e:
            self.errors.record('takeoff', e, log_level=40)
            return False
        finally:
            self.takeoff_in_progress = False
            self.last_cmd_time = time.monotonic()

    def _execute_velocity(self, vx, vy, vz, yaw_rate):
        mavutil = self._mavutil
        with self._tx_lock:
            self.master.mav.set_position_target_local_ned_send(
                0, self.target_system, self.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                0b0000011111000111,          # velocity + yaw-rate only
                0, 0, 0, vx, vy, vz, 0, 0, 0, 0, yaw_rate)

    def rtl(self) -> bool:
        log.warning("Commanding RTL.")
        self.hover()
        self._set_mode(ARDUCOPTER_MODE_RTL)
        return True

    def land(self) -> bool:
        log.warning("Commanding LAND.")
        self.hover()
        self._set_mode(ARDUCOPTER_MODE_LAND)
        return True


# ============================================================================
# PX4 — not implemented. Fail loudly instead of pretending to fly.
# ============================================================================
class PX4Backend(FCBackend):
    def _connect_impl(self):
        raise NotImplementedError("PX4 backend is not implemented yet (planned).")
    def _stream_telemetry(self): raise NotImplementedError
    def _execute_velocity(self, *a): raise NotImplementedError
    def get_telemetry(self): raise NotImplementedError
    def telemetry_age(self): raise NotImplementedError
    def consume_rc_events(self): raise NotImplementedError
    def arm_and_takeoff(self, altitude_m): raise NotImplementedError
    def rtl(self): raise NotImplementedError
    def land(self): raise NotImplementedError


# ============================================================================
# FACTORY WRAPPER
# ============================================================================
class FCInterface:
    def __init__(self, fc_type: str, connection: str,
                 fc_cfg: Optional[dict] = None,
                 rc_cfg: Optional[dict] = None,
                 safety_cfg: Optional[dict] = None):
        backends = {'ardupilot': ArduPilotBackend, 'px4': PX4Backend}
        key = fc_type.lower()
        if key not in backends:
            raise ValueError(f"Unsupported FC '{fc_type}'. Choices: {list(backends)}")
        self.backend = backends[key](connection, fc_cfg, rc_cfg, safety_cfg)

    def connect(self) -> bool:                 return self.backend.connect()
    def get_telemetry(self) -> dict:           return self.backend.get_telemetry()
    def telemetry_age(self):                   return self.backend.telemetry_age()
    def consume_rc_events(self) -> tuple:      return self.backend.consume_rc_events()
    def arm_and_takeoff(self, alt) -> bool:    return self.backend.arm_and_takeoff(alt)
    def send_velocity(self, vx, vy, vz, yaw):  self.backend.send_velocity(vx, vy, vz, yaw)
    def hover(self):                           self.backend.hover()
    def rtl(self) -> bool:                     return self.backend.rtl()
    def land(self) -> bool:                    return self.backend.land()
    def abort_takeoff(self):                   self.backend.abort_takeoff.set()
    def disconnect(self):                      self.backend.disconnect()

    @property
    def errors(self):                          return self.backend.errors


if __name__ == '__main__':
    import logging, yaml
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = yaml.safe_load(open('config.yaml'))
    fc = FCInterface('ardupilot', '127.0.0.1:14550',
                     cfg['fc'], cfg['rc'], cfg['safety'])
    if not fc.connect():
        raise SystemExit("connect failed")
    print("Monitoring RC for 30 s. In MAVProxy: rc 7 2000 / rc 8 2000 / rc 9 1000|1500|2000")
    end = time.time() + 30
    while time.time() < end:
        l, r, f = fc.consume_rc_events()
        if l: print("[TEST] LOCK event")
        if r: print("[TEST] RELEASE event")
        t = fc.get_telemetry()
        print(f"  filter={f} alt={t['alt']:.1f} armed={t['armed']} mode={t['mode']} "
              f"age={fc.telemetry_age():.2f}s", end='\r')
        time.sleep(0.2)
    fc.disconnect()
