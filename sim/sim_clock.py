"""Gazebo simulation clock and real-time factor (RTF).

If the simulator runs slower than real time (RTF < 1: GPU shared by Gazebo and the detector), anything moved by
the wall clock moves too fast for the physics: at RTF 0.5 a '5 m/s' target is 10 m/s in the drone's world.
Targets must follow SIM time. Needs python3-gz-transport13 / python3-gz-msgs10 (same as the camera source).
"""
import threading
import time


def sim_seconds(msg) -> float:
    st = msg.sim_time
    return float(st.sec) + float(st.nsec) * 1e-9


class SimClock:
    def __init__(self, world="tracker_world", wall=time.monotonic):
        from gz.msgs10.world_stats_pb2 import WorldStatistics
        from gz.transport13 import Node
        self._wall = wall
        self._node = Node()
        self._lock = threading.Lock()
        self._sim, self._wall_at = None, None
        self._hist = []                         # (wall, sim) pairs for RTF
        self.ok = self._node.subscribe(WorldStatistics, f"/world/{world}/stats", self._cb)

    def _cb(self, msg):
        s, w = sim_seconds(msg), self._wall()
        with self._lock:
            self._sim, self._wall_at = s, w
            self._hist.append((w, s))
            self._hist = [p for p in self._hist if w - p[0] <= 10.0]

    def now(self):
        """Sim seconds (interpolated between messages), or None if the stats topic is silent."""
        with self._lock:
            if self._sim is None or self._wall() - self._wall_at > 2.0:
                return None
            return self._sim + min(self._wall() - self._wall_at, 0.1)     # never run ahead of the sim by more than 0.1 s

    def rtf(self):
        with self._lock:
            if len(self._hist) < 2:
                return None
            (w0, s0), (w1, s1) = self._hist[0], self._hist[-1]
            return (s1 - s0) / (w1 - w0) if w1 > w0 else None
