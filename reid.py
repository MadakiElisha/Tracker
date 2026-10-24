"""Re-identification: pick the locked target back up after the tracker loses it.

ByteTrack gives a NEW id when a target is missed for more than a few frames (a fast car shrinking
into the distance, a detector dropout burst). Without this module the lock was simply dropped.

While the locked target is missing, every new track is scored against what we remembered:
  appearance (colour histogram of the box), where the target should be now (last position plus image
  velocity, gate widening with time), and size (the target shrinks when the drone falls behind).
A candidate is accepted only when it scores well, clearly beats the runner-up, and holds the best
score for `confirm_frames` frames in a row. Ambiguity means "keep searching", never "guess".
Tracks that already existed while the target was visible are other objects and are never candidates.
"""
import math
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

DEFAULTS = dict(
    enabled=True,
    timeout_s=6.0,            # give up (release the lock) after this long
    confirm_frames=3,         # consecutive frames the same candidate must win
    threshold=0.55,           # minimum combined score
    min_appearance=0.30,      # minimum colour similarity on its own
    margin=0.08,              # best must beat the runner-up by this much
    size_ratio_max=4.0,       # candidate area within [1/x, x] of the last seen area
    gate_min_frac=0.25,       # search radius floor, as a fraction of the frame width
    gate_growth_per_s=0.15,   # radius growth per second lost, fraction of frame width
    w_appearance=0.5, w_position=0.3, w_size=0.2,
)


def _hist(frame, box) -> Optional[np.ndarray]:
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.shape[0] > 32 or crop.shape[1] > 32:
        crop = cv2.resize(crop, (32, 32), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    s = hist.sum()
    return hist / s if s > 0 else None


def similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    """1 = identical colour distribution, 0 = disjoint (Bhattacharyya coefficient)."""
    if a is None or b is None:
        return 0.5                       # unknown: neither helps nor hurts much
    return float(np.sum(np.sqrt(a * b)))


class ReID:
    def __init__(self, cfg: Optional[dict] = None, frame_w: int = 640, frame_h: int = 480,
                 class_groups: Optional[Dict[str, List[int]]] = None, clock=time.monotonic):
        c = dict(DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.enabled = bool(c['enabled'])
        self.w, self.h = frame_w, frame_h
        self._clock = clock
        self._groups = [set(v) for k, v in (class_groups or {}).items() if k != 'any']
        self._first_seen: Dict[int, float] = {}
        self.reset()

    # -- bookkeeping ---------------------------------------------------------------------
    def reset(self):
        self._mem = None                 # dict, see remember()
        self._streak_id = None
        self._streak = 0
        self.last_event = ""

    @property
    def timeout_s(self) -> float:
        return float(self.cfg['timeout_s'])

    def observe_frame(self, tracks, t: Optional[float] = None):
        """Call every frame with all tracks so we know which ids pre-date a loss."""
        t = self._clock() if t is None else t
        for tr in tracks:
            self._first_seen.setdefault(int(tr[4]), t)
        if len(self._first_seen) > 2000:
            for k in sorted(self._first_seen, key=self._first_seen.get)[:1000]:
                del self._first_seen[k]

    def remember(self, frame, track, t: Optional[float] = None):
        """Call every frame the locked target is visible."""
        t = self._clock() if t is None else t
        x1, y1, x2, y2 = (float(v) for v in track[:4])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        m = self._mem
        vx = vy = 0.0
        if m is not None and t - m['t'] >= 0.03:
            dt = t - m['t']
            vx, vy = (cx - m['cx']) / dt, (cy - m['cy']) / dt
            vx, vy = 0.7 * m['vx'] + 0.3 * vx, 0.7 * m['vy'] + 0.3 * vy
        elif m is not None:
            vx, vy = m['vx'], m['vy']
        sig = _hist(frame, track)
        if m is not None and m['sig'] is not None and sig is not None:
            sig = 0.8 * m['sig'] + 0.2 * sig
            sig = sig / sig.sum()
        elif sig is None and m is not None:
            sig = m['sig']
        self._mem = dict(t=t, cx=cx, cy=cy, w=x2 - x1, h=y2 - y1, area=max((x2 - x1) * (y2 - y1), 1.0),
                         vx=vx, vy=vy, sig=sig, cls=int(track[6]), id=int(track[4]),
                         seen_ids={i for i, ts in self._first_seen.items() if ts <= t})
        self._streak_id, self._streak = None, 0

    def lost_for(self, t: Optional[float] = None) -> float:
        if self._mem is None:
            return 0.0
        return (self._clock() if t is None else t) - self._mem['t']

    def _same_group(self, a: int, b: int) -> bool:
        if a == b:
            return True
        return any(a in g and b in g for g in self._groups)

    # -- the search ----------------------------------------------------------------------
    def score(self, frame, track, t: float) -> Optional[dict]:
        """Score one candidate against memory; None if it is not eligible at all."""
        m = self._mem
        if m is None or int(track[4]) in m['seen_ids'] or not self._same_group(int(track[6]), m['cls']):
            return None
        c = self.cfg
        x1, y1, x2, y2 = (float(v) for v in track[:4])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dt = max(0.0, t - m['t'])
        px = m['cx'] + max(-0.5 * self.w, min(0.5 * self.w, m['vx'] * dt))
        py = m['cy'] + max(-0.5 * self.h, min(0.5 * self.h, m['vy'] * dt))
        radius = max(c['gate_min_frac'] * self.w, 2.0 * max(m['w'], m['h'])) + c['gate_growth_per_s'] * self.w * dt
        dist = math.hypot(cx - px, cy - py)
        if dist > radius:
            return None
        ratio = max((x2 - x1) * (y2 - y1), 1.0) / m['area']
        if not (1.0 / c['size_ratio_max'] <= ratio <= c['size_ratio_max']):
            return None
        pos = 1.0 - dist / radius
        size = 1.0 - abs(math.log(ratio)) / math.log(c['size_ratio_max'])
        app = similarity(m['sig'], _hist(frame, track))
        total = c['w_appearance'] * app + c['w_position'] * pos + c['w_size'] * size
        return dict(id=int(track[4]), score=total, app=app, pos=pos, size=size)

    def propose(self, frame, tracks, t: Optional[float] = None):
        """Return the track to re-lock to, or None. Call every frame while the lock is missing."""
        if not self.enabled or self._mem is None:
            return None
        t = self._clock() if t is None else t
        if t - self._mem['t'] > self.timeout_s:
            return None
        scored = []
        for tr in tracks:
            s = self.score(frame, tr, t)
            if s is not None:
                scored.append((s, tr))
        scored.sort(key=lambda x: -x[0]['score'])
        ok = (scored and scored[0][0]['score'] >= self.cfg['threshold']
              and scored[0][0]['app'] >= self.cfg['min_appearance']
              and (len(scored) == 1 or scored[0][0]['score'] - scored[1][0]['score'] >= self.cfg['margin']))
        if not ok:
            self._streak_id, self._streak = None, 0
            return None
        best, tr = scored[0]
        if self._streak_id == best['id']:
            self._streak += 1
        else:
            self._streak_id, self._streak = best['id'], 1
        if self._streak < self.cfg['confirm_frames']:
            return None
        self.last_event = (f"id {self._mem['id']}->{best['id']} score {best['score']:.2f} "
                           f"(appearance {best['app']:.2f}, position {best['pos']:.2f}, size {best['size']:.2f}) "
                           f"after {t - self._mem['t']:.1f}s")
        self._streak_id, self._streak = None, 0
        return tr
