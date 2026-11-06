#!/usr/bin/env python3
"""Why did 'distance within ±20%' fail? Per-lock range-error deciles, command saturation and a trace.

    python3 scripts/range_report.py logs/runs/<file>.csv [--cap 4.0]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_run as ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--cap", type=float, default=4.0, help="forward speed cap used (m/s)")
    ap.add_argument("--trace", type=int, default=25, help="rows of trace per lock")
    a = ap.parse_args()
    rows = ar.load(a.csv)
    for lid, t0, t1, n in ar.lock_segments(rows):
        seg = [r for r in rows if r["locked_id"] == lid and t0 <= ar._f(r["t"]) <= t1
               and r["target_visible"] in ("1", "True", "true")]
        if len(seg) < 30:
            continue
        re_ = sorted(ar._range_err(r) for r in seg)
        q = lambda p: re_[min(len(re_) - 1, int(p * len(re_)))]
        sat = sum(abs(ar._f(r["cmd_vx"])) >= 0.98 * a.cap for r in seg) / len(seg)
        far = sum(ar._range_err(r) > 0.2 for r in seg) / len(seg)
        near = sum(ar._range_err(r) < -0.2 for r in seg) / len(seg)
        print(f"\nlock {lid}: {t0:.0f}-{t1:.0f}s, {len(seg)} visible frames, setpoints {sorted({r['setpoint'] for r in seg})}")
        print("  range err deciles (+ = too far): " + " ".join(f"{q(p/10):+.2f}" for p in range(0, 10)) + f" max {re_[-1]:+.2f}")
        print(f"  too far >20%: {far:.0%}   too close >20%: {near:.0%}   cmd_vx at cap: {sat:.0%}")
        step = max(1, len(seg) // a.trace)
        print("  t     err_x  err_y  area    rangeErr cmd_vx cmd_vz cmd_yaw")
        for r in seg[::step][:a.trace]:
            print(f"  {ar._f(r['t']):5.1f} {ar._f(r['err_x']):+.2f}  {ar._f(r['err_y']):+.2f}  {ar._f(r['area']):.4f}  "
                  f"{ar._range_err(r):+.2f}    {ar._f(r['cmd_vx']):+.2f}  {ar._f(r['cmd_vz']):+.2f}  {ar._f(r['cmd_yaw']):+.2f}")


if __name__ == "__main__":
    main()
