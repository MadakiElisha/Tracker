"""Pure timeline maths for the scripted exhibition (no Gazebo needed; unit-tested).

A scenario is a list of segments. In each segment ONE actor (the sprinter or the car) is placed ahead of the
previous one, stands still while the drone locks it, then runs for `run_s` seconds. All other actors are hidden.
"""
from typing import Dict, List, Tuple

import highway_paths as hp

ACTORS = {            # name in the world, lane centre y, default lock class for the tracker
    "person": ("highway_runner", 8.5, "person"),
    "car":    ("highway_car", -1.75, "vehicle"),
}
HIDDEN_Z = -50.0


def load_segments(doc: dict) -> List[dict]:
    segs = []
    for i, s in enumerate(doc.get("segments") or []):
        actor = s.get("actor")
        if actor not in ACTORS:
            raise ValueError(f"segment {i + 1}: actor must be one of {sorted(ACTORS)}, got {actor!r}")
        speed = float(s["speed"])
        run_s = float(s["run_s"])
        if speed <= 0 or run_s <= 0:
            raise ValueError(f"segment {i + 1}: speed and run_s must be positive")
        segs.append(dict(
            name=str(s.get("name", f"Segment {i + 1}")), actor=actor, speed=speed, run_s=run_s,
            accel=float(s.get("accel", 3.0 if actor == "person" else 2.5)),
            lane_change_s=float(s.get("lane_change_s", 0.0)),
            vanish_after_s=float(s.get("vanish_after_s", 0.0)), vanish_for_s=float(s.get("vanish_for_s", 1.5)),
            note=str(s.get("note", "")), profile=s.get("profile"),
        ))
    if not segs:
        raise ValueError("scenario has no segments")
    gap = float(doc.get("gap_m", 10.0))
    x = hp.ROAD_START_X
    for s in segs:
        s["x0"] = x
        s["distance"] = hp.distance_travelled(s["run_s"], s["speed"], s["accel"])
        x = x + s["distance"] + gap
    if segs[-1]["x0"] + segs[-1]["distance"] > hp.ROAD_END_X:
        raise ValueError(f"scenario needs {segs[-1]['x0'] + segs[-1]['distance']:.0f} m of road but the road ends at "
                         f"{hp.ROAD_END_X:.0f} m: shorten run_s or lower speeds")
    return segs


def show_duration(segs: List[dict], lock_wait_s: float = 8.0) -> float:
    """Expected length of the moving part plus a typical lock wait per segment."""
    return sum(s["run_s"] + lock_wait_s for s in segs)


def actor_poses(seg: dict, t_move: float) -> Dict[str, Tuple[float, float, float, float]]:
    """{world name: (x, y, z, yaw)} for every actor, `t_move` seconds after the segment began to move
    (negative = still standing, waiting to be locked)."""
    out = {}
    for key, (name, y0, _cls) in ACTORS.items():
        if key != seg["actor"]:
            out[name] = (seg["x0"], y0, HIDDEN_Z, 0.0)
            continue
        t = max(0.0, t_move)
        d = hp.distance_travelled(t, seg["speed"], seg["accel"])
        y = y0 + (hp.lane_offset(t, 0.0, seg["lane_change_s"], 3.5) if seg["lane_change_s"] > 0 else 0.0)
        hidden = hp.vanished(t, 0.0, seg["vanish_after_s"], seg["vanish_for_s"]) if seg["vanish_after_s"] > 0 else False
        out[name] = (seg["x0"] + d, y, HIDDEN_Z if hidden else 0.0, 0.0)
    return out


PARKED = [f"hwy_parked_{i}" for i in range(4)]   # background SUVs on the shoulder (x from make_highway_world.py)
PARKED_X = (300, 700, 1100, 1500)


def parked_hidden_poses():
    """Poses that take the background traffic out of view so the tracker cannot lock onto it."""
    return {n: (float(x), -8.5, HIDDEN_Z, 0.0) for n, x in zip(PARKED, PARKED_X)}
