"""Stable class for a locked target.

The detector flips car/van/truck/bus frame to frame. The follow distance is set per class, so a flip
changes the setpoint (e.g. 0.004 -> 0.008) and the drone lurches. The class used for the setpoint is the
most common one seen on this lock over the last `window` frames (ties keep the current class).
"""
from collections import Counter, deque


class ClassVote:
    def __init__(self, window: int = 60):
        self.window = int(window)
        self._obs = deque(maxlen=self.window)
        self._id = None
        self._cur = None

    def update(self, locked_id, class_id):
        if locked_id != self._id:                 # new lock: start over
            self._id, self._cur = locked_id, None
            self._obs.clear()
        if class_id is None:
            return self._cur
        self._obs.append(int(class_id))
        counts = Counter(self._obs)
        best = max(counts.values())
        if self._cur is None or counts.get(self._cur, 0) < best:
            self._cur = next(c for c in reversed(self._obs) if counts[c] == best)
        return self._cur

    def rekey(self, new_id):
        """Carry the vote over when the lock moves to a re-identified track."""
        self._id = new_id
