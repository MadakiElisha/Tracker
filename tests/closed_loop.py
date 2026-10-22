#!/usr/bin/env python3
"""
=============================================================
 Closed-loop kinematic model of the follow-me loop.

 NOT a replacement for Gazebo/SITL — it models an idealised drone —
 but it exercises the real TrackerPID + limit_vertical against a
 pinhole camera, finite vehicle response, detection latency, pixel noise
 and dropped detections. It catches: wrong signs, saturation, oscillation,
 targets leaving the frame, and altitude-limit violations, in seconds and
 deterministically.

   python3 tests/closed_loop.py            # run all scenarios, print metrics
=============================================================
"""
import math
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pid_controller2 import TrackerPID          # noqa: E402
from safety import limit_vertical               # noqa: E402

W, H = 640, 480
HFOV_DEG = 90.0
F_PX = (W / 2) / math.tan(math.radians(HFOV_DEG / 2))


@dataclass
class Scenario:
    name: str
    target_path: Callable[[float], Tuple[float, float]]   # t -> (x, y) on ground
    duration_s: float = 60.0
    start: Tuple[float, float, float, float] = (0.0, 0.0, 10.0, 0.0)  # x,y,z,psi
    target_h: float = 1.7        # m
    target_w: float = 0.5        # m
    class_id: Optional[int] = 0
    cam_pitch_deg: float = 20.0
    latency_frames: int = 3      # detector + tracker latency
    pixel_noise: float = 2.0
    dropout: float = 0.02
    fps: float = 30.0
    seed: int = 1
    # vehicle response
    tau_v: float = 0.5
    tau_w: float = 0.3
    min_alt: float = 3.0
    max_alt: float = 30.0
    # airframe realism (defaults keep the original ideal behaviour)
    max_speed: float = 1e9           # m/s, horizontal airframe limit
    max_accel: float = 1e9           # m/s^2, horizontal acceleration limit
    tilt_coupling: bool = False      # camera pitch follows body pitch (fixed camera)
    gimbal: bool = False             # camera stays at cam_pitch_deg regardless of body tilt
    drag_tilt_deg_per_ms: float = 1.2   # steady-speed lean from drag, degrees per m/s
    tilt_comp: bool = False          # pass measured body tilt to the controller (tilt compensation)
    flight_cap: float = 1e9          # speed cap applied on the command (like flight.max_forward_speed)


@dataclass
class Result:
    t: List[float] = field(default_factory=list)
    err_x: List[float] = field(default_factory=list)
    err_y: List[float] = field(default_factory=list)
    range_err: List[float] = field(default_factory=list)   # (d - d_set)/d_set, true
    visible: List[bool] = field(default_factory=list)
    alt: List[float] = field(default_factory=list)
    yaw_cmd: List[float] = field(default_factory=list)
    dist: List[float] = field(default_factory=list)
    speed: List[float] = field(default_factory=list)
    tilt_deg: List[float] = field(default_factory=list)
    cmd_vx: List[float] = field(default_factory=list)
    cmd_vz: List[float] = field(default_factory=list)


def project(cam, psi, pitch, p):
    """World point -> (u, v, depth). x east, y north, z up; psi CCW from +x."""
    cx, cy, cz = cam
    fwd = (math.cos(psi) * math.cos(pitch), math.sin(psi) * math.cos(pitch), -math.sin(pitch))
    right = (math.sin(psi), -math.cos(psi), 0.0)
    up = (math.cos(psi) * math.sin(pitch), math.sin(psi) * math.sin(pitch), math.cos(pitch))
    v = (p[0] - cx, p[1] - cy, p[2] - cz)
    depth = sum(a * b for a, b in zip(v, fwd))
    if depth < 0.3:
        return None
    u = W / 2 + F_PX * sum(a * b for a, b in zip(v, right)) / depth
    w = H / 2 - F_PX * sum(a * b for a, b in zip(v, up)) / depth
    return u, w, depth


def bbox(cam, psi, pitch, tx, ty, th, tw, rng, noise):
    head = project(cam, psi, pitch, (tx, ty, th))
    feet = project(cam, psi, pitch, (tx, ty, 0.0))
    if head is None or feet is None:
        return None
    depth = (head[2] + feet[2]) / 2
    cxp = (head[0] + feet[0]) / 2
    half = F_PX * (tw / 2) / depth
    x1, x2 = cxp - half, cxp + half
    y1, y2 = head[1], feet[1]
    if noise:
        x1 += rng.gauss(0, noise); x2 += rng.gauss(0, noise)
        y1 += rng.gauss(0, noise); y2 += rng.gauss(0, noise)
    # must be at least partly in frame and bigger than a few pixels
    ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
    if not (0 <= ccx <= W and 0 <= ccy <= H) or (x2 - x1) < 3 or (y2 - y1) < 3:
        return None
    return (max(0, x1), max(0, y1), min(W, x2), min(H, y2))


def run(scn: Scenario, pid_cfg: dict, setpoint_area: Optional[float] = None) -> Result:
    rng = random.Random(scn.seed)
    dt = 1.0 / scn.fps
    now = [0.0]
    pid = TrackerPID.from_config(pid_cfg, clock=lambda: now[0])
    if setpoint_area is not None:
        pid.setpoint_area = setpoint_area
        pid.setpoint_by_class = {}
    x, y, z, psi = scn.start
    vx = vz = wz = 0.0
    pitch0 = math.radians(scn.cam_pitch_deg)
    pitch = pitch0
    ax = 0.0
    cmd = (0.0, 0.0, 0.0)
    hist: List[Tuple[float, float, float, float, float, float]] = []
    pitch_hist: List[float] = []
    tilt_hist: List[float] = []
    tilt = 0.0
    res = Result()
    sp = pid.setpoint_for(scn.class_id)

    steps = int(scn.duration_s * scn.fps)
    for i in range(steps):
        t = i * dt
        now[0] = t
        tx, ty = scn.target_path(t)
        hist.append((x, y, z, psi, tx, ty))
        pitch_hist.append(pitch)
        tilt_hist.append(tilt)
        # sensing uses the state `latency` frames ago
        lag = max(0, len(hist) - 1 - scn.latency_frames)
        sx, sy, sz, spsi, stx, sty = hist[lag]
        box = None
        if rng.random() >= scn.dropout:
            box = bbox((sx, sy, sz), spsi, pitch_hist[lag], stx, sty,
                       scn.target_h, scn.target_w, rng, scn.pixel_noise)
        if box is not None:
            cmd = pid.compute(box, W, H, class_id=scn.class_id,
                              tilt_rad=(0.0 if (scn.gimbal or not scn.tilt_comp) else tilt_hist[lag]),
                              alt_m=sz, v_drone_fwd=vx)
        else:
            f = 0.5 ** (dt / 0.15)                      # same time-based decay as tracker.py
            cmd = (cmd[0] * f, cmd[1] * f, cmd[2] * f)
        cvx, cvz, cyaw = cmd
        cvx = max(-scn.flight_cap, min(scn.flight_cap, cvx))
        cvz = limit_vertical(cvz, z, scn.min_alt, scn.max_alt)

        # vehicle: first-order response with acceleration + speed limits, body-frame forward, NED vz (+down)
        ax = max(-scn.max_accel, min(scn.max_accel, (cvx - vx) / scn.tau_v))
        vx += ax * dt
        vx = max(-scn.max_speed, min(scn.max_speed, vx))
        vz += (cvz - vz) * dt / scn.tau_v
        wz += (-cyaw - wz) * dt / scn.tau_w           # +yaw cmd = clockwise = psi decreasing
        psi += wz * dt
        # body tilt: acceleration lean + drag lean (nose down = positive); a fixed camera follows it
        tilt = math.atan2(ax, 9.81) + math.radians(scn.drag_tilt_deg_per_ms) * vx
        pitch = pitch0 if (scn.gimbal or not scn.tilt_coupling) else pitch0 + tilt
        x += vx * math.cos(psi) * dt
        y += vx * math.sin(psi) * dt
        z -= vz * dt

        # ground truth for scoring (no latency/noise)
        tb = bbox((x, y, z), psi, pitch, tx, ty, scn.target_h, scn.target_w, rng, 0.0)
        res.t.append(t)
        res.alt.append(z)
        res.yaw_cmd.append(cyaw)
        res.speed.append(vx)
        res.tilt_deg.append(math.degrees(tilt))
        res.cmd_vx.append(cvx)
        res.cmd_vz.append(cvz)
        res.dist.append(math.hypot(tx - x, ty - y))
        if tb is None:
            res.visible.append(False)
            res.err_x.append(float('nan')); res.err_y.append(float('nan'))
            res.range_err.append(float('nan'))
        else:
            res.visible.append(True)
            res.err_x.append(((tb[0] + tb[2]) / 2) / W - 0.5)
            res.err_y.append(((tb[1] + tb[3]) / 2) / H - 0.5)
            a = (tb[2] - tb[0]) * (tb[3] - tb[1]) / (W * H)
            res.range_err.append(math.sqrt(sp / max(a, 1e-9)) - 1.0)
    return res


def metrics(res: Result, settle_s: float = 15.0) -> dict:
    idx = [i for i, t in enumerate(res.t) if t >= settle_s]
    vis = [i for i in idx if res.visible[i]]
    n = max(len(idx), 1)
    ex = [abs(res.err_x[i]) for i in vis]
    re_ = [abs(res.range_err[i]) for i in vis]
    ys = [res.yaw_cmd[i] for i in idx if abs(res.yaw_cmd[i]) > 0.05]
    flips = sum(1 for a, b in zip(ys, ys[1:]) if a * b < 0)
    span = max(res.t[-1] - settle_s, 1e-6)
    return {
        'visible_frac': len(vis) / n,
        'median_abs_err_x': sorted(ex)[len(ex) // 2] if ex else float('nan'),
        'median_abs_range_err': sorted(re_)[len(re_) // 2] if re_ else float('nan'),
        'p90_abs_range_err': sorted(re_)[int(len(re_) * 0.9)] if re_ else float('nan'),
        'yaw_flips_per_10s': flips / span * 10.0,
        'min_alt': min(res.alt),
        'max_alt': max(res.alt),
        'final_dist': res.dist[-1],
    }


# ---- target paths -----------------------------------------------------------
def stationary(x=15.0, y=0.0):
    return lambda t: (x, y)

def walk_away(speed=1.4, x0=15.0):
    return lambda t: (x0 + speed * t, 0.0)

def walk_across(speed=1.4, x0=15.0, y0=-10.0):
    return lambda t: (x0, y0 + speed * t)

def circle(radius=20.0, speed=1.4):
    w = speed / radius
    return lambda t: (radius * math.cos(w * t), radius * math.sin(w * t))

def stop_and_go(speed=1.5, x0=15.0, period=20.0):
    def f(t):
        done = (t // period) * period / 2 * speed + (min(t % period, period / 2) * speed)
        return (x0 + done, 0.0)
    return f


SCENARIOS = [
    Scenario("stationary 15 m ahead", stationary(), 60),
    Scenario("walking away 1.4 m/s", walk_away(1.4), 90),
    Scenario("crossing laterally 1.4 m/s", walk_across(1.4), 60),
    Scenario("walking circle r=20", circle(), 120, start=(25.0, 0.0, 10.0, math.pi / 2)),
    Scenario("stop-and-go 1.5 m/s", stop_and_go(1.5), 90),
]

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'config.yaml')))
    for s in SCENARIOS:
        m = metrics(run(s, cfg['pid']))
        print(f"{s.name:<30} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
