"""Pure target-motion maths for the 2 km highway scenarios (no Gazebo needed; unit-tested).

Everything is a function of time since the mover started. x runs along the road, y across it.
"""
import math

ROAD_START_X = 20.0       # targets start 20 m ahead of the drone (which spawns at x=0)
ROAD_END_X = 2000.0       # road is 2 km; targets stop here


def distance_travelled(t: float, speed: float, accel: float, delay: float = 0.0) -> float:
    """Metres covered at time t: wait `delay` s, ramp up at `accel` m/s^2 to `speed`, then cruise."""
    tt = max(0.0, t - delay)
    if tt == 0.0 or speed <= 0.0:
        return 0.0
    if accel <= 0.0:
        return speed * tt
    t_ramp = speed / accel
    if tt < t_ramp:
        return 0.5 * accel * tt * tt
    return 0.5 * speed * t_ramp + speed * (tt - t_ramp)


def smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def lane_offset(t: float, delay: float, every_s: float, width: float, change_s: float = 3.0) -> float:
    """Lateral offset: every `every_s` s the target moves one lane (`width` m) over `change_s` s, alternating."""
    if every_s <= 0.0 or width == 0.0:
        return 0.0
    tt = t - delay
    if tt <= 0.0:
        return 0.0
    n = int(tt // every_s)
    phase = smoothstep((tt - n * every_s) / change_s)
    # n completed changes alternate +w, 0, +w, 0 ...
    base = width * (n % 2)
    return base + (width * phase if n % 2 == 0 else -width * phase)


def pose(t: float, speed: float, accel: float, y0: float, delay: float = 0.0,
         lane_every_s: float = 0.0, lane_width: float = 3.5, x0: float = ROAD_START_X):
    """(x, y, yaw) of a target driving/running along +x, stopping at the end of the road."""
    d = distance_travelled(t, speed, accel, delay)
    x = min(x0 + d, ROAD_END_X)
    y = y0 + lane_offset(t, delay, lane_every_s, lane_width)
    return x, y, 0.0


def vanished(t: float, delay: float, every_s: float, for_s: float) -> bool:
    """True while a target should be hidden (to provoke a detector dropout / track-id change on demand)."""
    if every_s <= 0.0 or for_s <= 0.0:
        return False
    tt = t - delay
    return tt >= every_s and ((tt - every_s) % (every_s + for_s)) < for_s
