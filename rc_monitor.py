#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — RC Channel Monitor

 Reads RC_CHANNELS MAVLink messages on a dedicated thread
 and exposes debounced switch states for target assignment.

 Channel mapping (PWM ranges):
   CH7  < 1300        -> idle
   CH7  > 1700        -> LOCK_LARGEST  (edge-triggered)
   CH8  > 1700        -> RELEASE_LOCK  (edge-triggered)
   CH9  < 1300        -> filter ANY
   CH9  ~1500         -> filter PERSON
   CH9  > 1700        -> filter VEHICLE
=============================================================
"""

import time
import threading


class RCMonitor:
    def __init__(self, master, channels=None):
        """
        master: a pymavlink mavlink_connection object
                (re-use the same connection as ArduPilotBackend
                 — pass fc.backend.master)
        """
        self.master   = master
        self.running  = False
        self._thread  = None
        self._lock    = threading.Lock()

        # Raw channel values (1000-2000 typical)
        self.channels = {ch: 1500 for ch in range(1, 17)}

        # Edge-triggered event flags (consumed by tracker)
        self.lock_event    = False
        self.release_event = False

        # Last known states for edge detection
        self._prev_ch7_high = False
        self._prev_ch8_high = False

        # Class filter state: 'any' | 'person' | 'vehicle'
        self.class_filter = 'any'

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[RC_MONITOR] Started.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _loop(self):
        while self.running:
            try:
                msg = self.master.recv_match(
                    type='RC_CHANNELS', blocking=True, timeout=0.5)
                if msg is None:
                    continue

                with self._lock:
                    self.channels[1]  = msg.chan1_raw
                    self.channels[2]  = msg.chan2_raw
                    self.channels[3]  = msg.chan3_raw
                    self.channels[4]  = msg.chan4_raw
                    self.channels[5]  = msg.chan5_raw
                    self.channels[6]  = msg.chan6_raw
                    self.channels[7]  = msg.chan7_raw
                    self.channels[8]  = msg.chan8_raw
                    self.channels[9]  = msg.chan9_raw

                    ch7 = self.channels[7]
                    ch8 = self.channels[8]
                    ch9 = self.channels[9]

                    # --- CH7: Lock largest (rising edge) ---
                    ch7_high = ch7 > 1700
                    if ch7_high and not self._prev_ch7_high:
                        self.lock_event = True
                        print("[RC_MONITOR] CH7 HIGH -> LOCK requested.")
                    self._prev_ch7_high = ch7_high

                    # --- CH8: Release (rising edge) ---
                    ch8_high = ch8 > 1700
                    if ch8_high and not self._prev_ch8_high:
                        self.release_event = True
                        print("[RC_MONITOR] CH8 HIGH -> RELEASE requested.")
                    self._prev_ch8_high = ch8_high

                    # --- CH9: Class filter (3-position) ---
                    if ch9 < 1300:
                        new_filter = 'any'
                    elif ch9 > 1700:
                        new_filter = 'vehicle'
                    else:
                        new_filter = 'person'

                    if new_filter != self.class_filter:
                        self.class_filter = new_filter
                        print(f"[RC_MONITOR] Class filter -> {new_filter}")

            except Exception as e:
                print(f"[RC_MONITOR] Error: {e}")
                time.sleep(0.1)

    def consume_lock_event(self) -> bool:
        """Returns True once, then resets. Call every frame."""
        with self._lock:
            if self.lock_event:
                self.lock_event = False
                return True
            return False

    def consume_release_event(self) -> bool:
        with self._lock:
            if self.release_event:
                self.release_event = False
                return True
            return False

    def get_class_filter(self) -> str:
        with self._lock:
            return self.class_filter


if __name__ == '__main__':
    # Standalone test — connects directly and prints RC state
    from pymavlink import mavutil

    print("Connecting...")
    master = mavutil.mavlink_connection('udpin:127.0.0.1:14550')
    master.wait_heartbeat()
    print("Connected. Monitoring RC channels (Ctrl+C to stop)...")
    print("Use MAVProxy: 'rc 7 2000' to test lock, 'rc 7 1000' to reset")

    rc = RCMonitor(master)
    rc.start()

    try:
        while True:
            print(f"  CH7:{rc.channels[7]} CH8:{rc.channels[8]} "
                  f"CH9:{rc.channels[9]} filter:{rc.get_class_filter()}  "
                  f"lock_evt:{rc.lock_event} release_evt:{rc.release_event}",
                  end='\r')
            time.sleep(0.2)
    except KeyboardInterrupt:
        rc.stop()
        print("\nStopped.")
