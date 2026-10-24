#!/usr/bin/env python3
"""Find camera.sync_lag_s: how old the frame is compared with the telemetry it is matched to.

    python3 scripts/calibrate_sync.py logs/runs/<file>.csv [--hfov 90] [--width 640] [--height 480]

Why: the drone's lean moves a fixed camera's image. The controller removes that using the lean AT THE TIME THE
FRAME WAS TAKEN. A wrong lag of even 50 ms breaks fast chases (offline model), so measure it instead of guessing.

How: while a target is tracked its true elevation angle changes slowly, but the image position swings with the
lean. For each candidate lag L the corrected angle  atan(err_y*H/f) + lean(t - L)  is computed from the run
log; the right L is the one that makes it steadiest. Needs a run where the drone leans visibly (a chase, or
back-and-forth flying with a locked target). Put the result in config.yaml: camera.sync_lag_s.
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_run as ar


def _interp(ts, vs, t):
    import bisect
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return vs[0]
    if i >= len(ts):
        return vs[-1]
    t0, t1 = ts[i - 1], ts[i]
    w = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
    return vs[i - 1] + w * (vs[i] - vs[i - 1])


def _highpass(ts, vs, win=3.0):
    """Remove the slow trend (true elevation changing as the gap changes)."""
    out, j0, j1, acc = [], 0, 0, 0.0
    n = len(ts)
    for i in range(n):
        while j1 < n and ts[j1] <= ts[i] + win / 2:
            acc += vs[j1]
            j1 += 1
        while ts[j0] < ts[i] - win / 2:
            acc -= vs[j0]
            j0 += 1
        out.append(vs[i] - acc / (j1 - j0))
    return out


def estimate(rows, hfov_deg=90.0, width=640, height=480, max_lag=0.6, step=0.01, min_lean_deg=2.0):
    """Return dict(lag_s, improvement, lean_std_deg, n) or dict(error=...)."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    ts_all, lean_all = [], []
    for r in rows:
        if r.get("tilt_now_deg", "") != "" and r.get("t_read", "") != "":
            ts_all.append(ar._f(r["t_read"]))
            lean_all.append(math.radians(ar._f(r["tilt_now_deg"])))
    pts = []
    for r in rows:
        if (str(r.get("airborne")) in ("1", "True", "true") and r.get("locked_id", "") != ""
                and str(r.get("target_visible")) in ("1", "True", "true")
                and r.get("t_read", "") != "" and r.get("err_y", "") != ""):
            pts.append((ar._f(r["t_read"]), math.atan(ar._f(r["err_y"]) * height / f)))
    if len(ts_all) < 50 or len(pts) < 100:
        return dict(error="not enough tracked samples (need a run with a locked target and the t_read / tilt_now_deg columns)")
    mean = sum(lean_all) / len(lean_all)
    lean_std = math.degrees(math.sqrt(sum((x - mean) ** 2 for x in lean_all) / len(lean_all)))
    if lean_std < min_lean_deg:
        return dict(error=f"the drone barely leaned (std {lean_std:.1f} deg): fly a chase or back-and-forth first")
    t = [p[0] for p in pts]
    ang = [p[1] for p in pts]

    def score(lag):
        comp = [a + _interp(ts_all, lean_all, tt - lag) for tt, a in zip(t, ang)]
        hp = _highpass(t, comp)
        return sum(x * x for x in hp) / len(hp)

    n_steps = int(max_lag / step) + 1
    scores = [(score(k * step), k * step) for k in range(n_steps)]
    best_s, best_lag = min(scores)
    s0 = scores[0][0]
    none_s = _variance_no_comp(t, ang)
    return dict(lag_s=best_lag, improvement=1.0 - best_s / none_s if none_s > 0 else 0.0,
                at_zero=1.0 - s0 / none_s if none_s > 0 else 0.0, lean_std_deg=lean_std, n=len(pts))


def _variance_no_comp(t, ang):
    hp = _highpass(t, ang)
    return sum(x * x for x in hp) / len(hp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--hfov", type=float, default=90.0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    a = ap.parse_args()
    res = estimate(ar.load(a.csv), a.hfov, a.width, a.height)
    if "error" in res:
        print("Cannot calibrate:", res["error"])
        sys.exit(2)
    print(f"samples {res['n']}, lean std {res['lean_std_deg']:.1f} deg")
    print(f"image wobble removed with lag 0: {res['at_zero']:.0%}; with best lag: {res['improvement']:.0%}")
    print(f"\nRecommended:  camera.sync_lag_s: {res['lag_s']:.2f}")
    if res["improvement"] < 0.2:
        print("WARNING: tilt correction removes little of the wobble even at the best lag; do not trust this value.")
        sys.exit(1)


if __name__ == "__main__":
    main()
