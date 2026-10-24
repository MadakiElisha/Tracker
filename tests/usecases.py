"""
Use-case matrix for fast pursuit, run in the offline closed-loop model.

    python3 tests/usecases.py                 # current config.yaml
    python3 tests/usecases.py --profile fast  # a profile from config.yaml (see `profiles:`)

Each row: a target type + speed + manoeuvre, flown by an airframe with realistic limits
(speed, acceleration, body tilt that moves a fixed camera). This is a MODEL: it ranks
controller changes, it does not replace Gazebo or a real flight.
"""
import copy, dataclasses, math, os, sys
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import closed_loop as cl            # noqa: E402

Path = Callable[[float], Tuple[float, float]]


# ---- target path generators (x east, y north; drone starts at the origin facing +x) -------------
def line(v, x0=15.0, y0=0.0, hdg=0.0, hold_s=0.0, accel=None):
    """Straight run at v m/s along heading hdg; optionally stationary for hold_s, then ramping at accel."""
    def f(t):
        tt = max(0.0, t - hold_s)
        if accel:
            d = 0.5 * accel * tt * tt if tt < v / accel else v * v / (2 * accel) + v * (tt - v / accel)
        else:
            d = v * tt
        return x0 + d * math.cos(hdg), y0 + d * math.sin(hdg)
    return f


def stop_go(v, period=30.0, x0=15.0, accel=3.0):
    """Drive for period/2, stop for period/2, with finite accel/brake (a car cannot stop instantly)."""
    cache = {}

    def vel(t):
        ph = t % period
        half = period / 2
        ta = v / accel
        if ph < half:                      # driving phase: ramp up, cruise, ramp down at the end
            up = min(ph / ta, 1.0)
            dn = min((half - ph) / ta, 1.0)
            return v * min(up, dn)
        return 0.0

    def f(t):
        n = int(t / 0.01)
        if n not in cache:                 # integrate lazily, memoised
            last = max(cache) if cache else -1
            x = cache[last] if last >= 0 else x0
            for k in range(last + 1, n + 1):
                x += vel(k * 0.01) * 0.01
                cache[k] = x
        return cache[n], 0.0
    return f


def zigzag(v, amp=4.0, period=12.0, x0=15.0):
    return lambda t: (x0 + v * t, amp * math.sin(2 * math.pi * t / period))


def loop(v, radius, x0=15.0):
    w = v / radius
    return lambda t: (x0 + radius * math.sin(w * t), radius * (1 - math.cos(w * t)))


def patrol(v, amp=20.0, y0=-15.0):
    period = 2 * math.pi * amp / v
    return lambda t: (amp * math.sin(2 * math.pi * t / period), y0)


@dataclass
class UseCase:
    name: str
    path: Path
    class_id: int
    h: float
    w: float
    duration: float = 90.0
    expect: str = "pass"          # "pass" | "exceeds_airframe" (informational)
    start: Optional[Tuple[float, float, float, float]] = None   # x, y, z, psi override
    profile: str = "pedestrian"   # profile in config.yaml that this use case is flown with
    airframe: Optional[dict] = None   # overrides on AIRFRAME for this case (e.g. AIRFRAMES['quad_10_15in'])
    note: str = ""


# Airframe presets. quad_10_15in is an ASSUMED typical 10-15 inch frame (no measured data): fast, strong
# thrust, acceleration limited by the autopilot (WPNAV_ACCEL 600 cm/s^2 in ardupilot_fast.param).
AIRFRAMES = {
    "quad_10_15in": dict(max_speed=30.0, max_accel=6.0, drag_tilt_deg_per_ms=1.0),
}


PERSON = dict(class_id=0, h=1.7, w=0.5)
CAR = dict(class_id=3, h=1.5, w=2.0, profile="vehicle")
BUS = dict(class_id=8, h=3.0, w=2.5, profile="vehicle")
MOTO = dict(class_id=9, h=1.4, w=0.8, profile="motorcycle")

USE_CASES = [
    UseCase("pedestrian walking 1.4 m/s", line(1.4), **PERSON),
    UseCase("jogger 3 m/s", line(3.0), **PERSON),
    UseCase("sprinter 7 m/s (starts at rest)", line(7.0, hold_s=5, accel=3.0), duration=60, **{**PERSON, "profile": "runner"}),
    UseCase("cyclist 8 m/s", line(8.0, hold_s=5, accel=1.5), duration=70, **{**MOTO, 'class_id': 9}),
    UseCase("motorcycle 15 m/s", line(15.0, hold_s=5, accel=3.0), duration=60, **MOTO),
    UseCase("city car 12 m/s", line(12.0, hold_s=5, accel=2.5), duration=60, **CAR),
    UseCase("city car 12 m/s, lane changes", zigzag(12.0, amp=4.0, period=14.0), duration=60, **CAR),
    UseCase("car cornering 10 m/s, r=40 m", loop(10.0, 40.0), duration=70, **CAR),
    UseCase("traffic stop-and-go 8 m/s", stop_go(8.0, 30.0), duration=120, **CAR),
    UseCase("bus 10 m/s", line(10.0, hold_s=5, accel=1.5), duration=60, **BUS),
    UseCase("sim pickup patrol, 2.1 m/s peak", patrol(2.1), duration=120, start=(0.0, 0.0, 10.0, -math.pi / 2), **CAR),
    UseCase("highway car 28 m/s", line(28.0, hold_s=5, accel=3.0), duration=60,
            expect="exceeds_airframe", note="needs an airframe faster than the modelled one", **CAR),
    # --- long straight road (2 km), assumed 10-15 inch quad, fixed camera, runs > 60 s after settling ---
    UseCase("highway sprinter 10 m/s", line(10.0, hold_s=5, accel=3.0), duration=100,
            profile="highway_runner", airframe=AIRFRAMES["quad_10_15in"], note="2 km road", **{k: v for k, v in PERSON.items()}),
    UseCase("highway car 20 m/s", line(20.0, hold_s=5, accel=3.0), duration=120,
            airframe=AIRFRAMES["quad_10_15in"], note="2 km road", **{**CAR, "profile": "highway"}),
]

AIRFRAME = dict(max_speed=20.0, max_accel=5.0, tilt_coupling=True, tilt_comp=True, tau_v=0.5, latency_frames=3,
                fps=30.0, pixel_noise=2.0, dropout=0.02, seed=1)


def evaluate(uc: UseCase, cfg: dict, gimbal=None, airframe=None, extra=None):
    """Fly one use case with its mission profile merged onto cfg. gimbal=None -> profile's camera.gimbal."""
    from profiles import apply_profile
    c = apply_profile(cfg, uc.profile)
    pid_cfg = c['pid']
    flight_cap = c['flight']['max_forward_speed']
    alt = c['flight'].get('target_altitude', 10.0)
    if gimbal is None:
        gimbal = bool(c.get('camera', {}).get('gimbal', False))
    kw = dict(AIRFRAME, **(uc.airframe or {}), **(airframe or {}))
    scn = cl.Scenario(uc.name, uc.path, duration_s=uc.duration, class_id=uc.class_id,
                      target_h=uc.h, target_w=uc.w, flight_cap=flight_cap, gimbal=gimbal, **kw)
    scn = dataclasses.replace(scn, start=uc.start or (0.0, 0.0, alt, 0.0))
    if extra:
        scn = dataclasses.replace(scn, **extra)
    res = cl.run(scn, pid_cfg)
    settle = 15.0
    idx = [i for i, t in enumerate(res.t) if t >= settle]
    n = max(len(idx), 1)
    vis = [i for i in idx if res.visible[i]]
    within = sum(1 for i in vis if abs(res.range_err[i]) <= 0.20) / n
    re_ = sorted(abs(res.range_err[i]) for i in vis)
    return dict(visible=len(vis) / n, within20=within,
                med_re=re_[len(re_) // 2] if re_ else float('nan'),
                max_speed=max(res.speed), max_tilt=max(abs(x) for x in res.tilt_deg),
                final_dist=res.dist[-1])


def verdict(m):
    return "PASS" if (m['visible'] >= 0.98 and m['within20'] >= 0.70) else "FAIL"


def load_cfg():
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return yaml.safe_load(open(os.path.join(root, 'config.yaml')))


def run_matrix(cfg=None, gimbal=None, airframe=None):
    cfg = cfg or load_cfg()
    return [(uc, evaluate(uc, cfg, gimbal=gimbal, airframe=airframe)) for uc in USE_CASES]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--gimbal', choices=['profile', 'on', 'off'], default='profile',
                    help="profile = use each profile's camera.gimbal")
    a = ap.parse_args()
    g = {'profile': None, 'on': True, 'off': False}[a.gimbal]
    print(f"{'use case':36s} {'profile':10s} {'verdict':7s} visible within20 medRE  maxV  tilt  final_d")
    npass = 0
    for uc, m in run_matrix(gimbal=g):
        v = verdict(m); npass += v == "PASS"
        print(f"{uc.name:36s} {uc.profile:10s} {v:7s} {m['visible']:6.2f} {m['within20']:8.2f} {m['med_re']:6.2f} "
              f"{m['max_speed']:5.1f} {m['max_tilt']:4.0f}° {m['final_dist']:7.1f}  {uc.note}")
    print(f"PASS {npass}/{len(USE_CASES)}")


if __name__ == '__main__':
    main()
