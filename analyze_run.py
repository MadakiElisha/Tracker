#!/usr/bin/env python3
"""
=============================================================
 Drone Tracker — Run Analyzer (sim/flight validation scorer)

 Scores a CSV written by run_logger.py against the pass/fail
 criteria in docs/SIM_VALIDATION.md.

   python3 analyze_run.py logs/runs/run_*.csv
   python3 analyze_run.py run.csv --follow-seconds 300 --soak-seconds 1800

 Each check reports PASS, FAIL or INCONCLUSIVE (the run did not
 exercise that situation). Exit code is 1 if any check FAILs.
 INCONCLUSIVE is not a pass: it means "go provoke that case".
=============================================================
"""
import argparse
import csv
import sys
from statistics import median

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

REQUIRED = ["t", "airborne", "mode", "locked_id", "target_visible",
            "err_x", "err_y", "area", "setpoint", "cmd_vx", "cmd_vz",
            "cmd_yaw", "safety_action", "safety_reason", "errors", "fps"]


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _cmd_mag(r):
    return max(abs(_f(r["cmd_vx"])), abs(_f(r["cmd_vz"])), abs(_f(r["cmd_yaw"])))


def _tracking_rows(rows):
    return [r for r in rows if r["airborne"] in ("1", "True", "true")
            and r["locked_id"] != "" and r["target_visible"] in ("1", "True", "true")]


def _tracked_seconds(rows):
    total = 0.0
    for a, b in zip(rows, rows[1:]):
        dt = _f(b["t"]) - _f(a["t"])
        if 0 < dt < 1.0:
            total += dt
    return total


# --------------------------------------------------------------------------
def check_log_integrity(rows):
    if not rows:
        return FAIL, "log is empty"
    missing = [c for c in REQUIRED if c not in rows[0]]
    if missing:
        return FAIL, f"missing columns: {missing}"
    ts = [_f(r["t"]) for r in rows]
    if any(b < a for a, b in zip(ts, ts[1:])):
        return FAIL, "timestamps not monotonic"
    return PASS, f"{len(rows)} rows, {ts[-1]:.1f}s"


def check_follow(rows, min_seconds, settle_s=10.0,
                 max_yaw_flips_per_10s=10.0):
    tr = _tracking_rows(rows)
    secs = _tracked_seconds(tr)
    if secs < min_seconds:
        return INCONCLUSIVE, f"only {secs:.0f}s tracked while airborne (need {min_seconds:.0f}s)"
    t0 = _f(tr[0]["t"]) + settle_s
    body = [r for r in tr if _f(r["t"]) >= t0]
    if len(body) < 30:
        return INCONCLUSIVE, "too few samples after settle time"
    centred = sum(abs(_f(r["err_x"])) < 0.15 and abs(_f(r["err_y"])) < 0.20
                  for r in body) / len(body)
    # "within +-20% of the desired distance": relative RANGE error (what the PID controls),
    # not area (area error is ~2x the range error, so +-20% area would be ~+-10% range).
    dist_ok = sum(
        _f(r["setpoint"]) > 0 and _f(r["area"]) > 0 and abs(_range_err(r)) <= 0.20
        for r in body) / len(body)
    # oscillation: sign flips of meaningful yaw commands
    ys = [_f(r["cmd_yaw"]) for r in body if abs(_f(r["cmd_yaw"])) > 0.05]
    flips = sum(1 for a, b in zip(ys, ys[1:]) if a * b < 0)
    span = max(_f(body[-1]["t"]) - _f(body[0]["t"]), 1e-6)
    flip_rate = flips / span * 10.0
    detail = (f"centred {centred:.0%}, distance within ±20% {dist_ok:.0%}, "
              f"yaw flips {flip_rate:.1f}/10s")
    ok = centred >= 0.90 and dist_ok >= 0.80 and flip_rate <= max_yaw_flips_per_10s
    return (PASS if ok else FAIL), detail


RANGE_DEADBAND = 0.05      # = pid.pitch.deadzone (relative range error)


def _range_err(r):
    """Same error the pitch loop acts on: sqrt(setpoint/area) - 1 (positive = too far)."""
    area = max(_f(r["area"]), 1e-6)
    return (max(_f(r["setpoint"]), 1e-6) / area) ** 0.5 - 1.0


def check_no_creep(rows, rest_window=None):
    """Creep = motion commanded while the TARGET IS STATIONARY and already inside every deadband.
    The log cannot tell a parked target from one moving at constant speed (a drone following a
    walker must keep commanding forward speed, with the integral term holding it), so the check
    only judges a window the operator says the target was at rest: --rest-window START END."""
    if rest_window is None:
        return INCONCLUSIVE, ("needs a stationary target: lock a parked car / standing person for ~60 s "
                              "and pass --rest-window START END (seconds in the log)")
    lo, hi = rest_window
    rows = [r for r in rows if lo <= _f(r["t"]) <= hi]
    tr = _tracking_rows(rows)
    still = [r for r in tr
             if abs(_f(r["err_x"])) < 0.03 and abs(_f(r["err_y"])) < 0.04
             and abs(_range_err(r)) < RANGE_DEADBAND]
    if len(still) < 30:
        return INCONCLUSIVE, "target never held steady inside the deadbands"
    moving = [r for r in still if _cmd_mag(r) > 0.08]
    creeping = len(moving) / len(still)
    detail = f"{creeping:.1%} of in-deadband frames commanded motion (limit 5%)"
    if moving:
        axis = {k: sum(abs(_f(r[k])) > 0.08 for r in moving) for k in ("cmd_vx", "cmd_vz", "cmd_yaw")}
        detail += f"; offending axes (frames): {axis}"
    return (PASS if creeping < 0.05 else FAIL), detail


def check_lock_loss(rows, max_lost_frames=60, decay_s=1.0):
    air = [r for r in rows if r["airborne"] in ("1", "True", "true")]
    runs, cur = [], None
    for r in air:
        invisible = r["target_visible"] in ("0", "False", "false")
        if cur is None and r["locked_id"] != "" and invisible:
            cur = [r]
        elif cur is not None and invisible:
            cur.append(r)
        elif cur is not None:
            runs.append(cur); cur = None
    if cur:
        runs.append(cur)
    runs = [x for x in runs if _f(x[-1]["t"]) - _f(x[0]["t"]) >= decay_s]
    if not runs:
        return INCONCLUSIVE, "locked target never disappeared long enough — hide it behind the wall"
    fps = median([_f(r["fps"], 15.0) for r in air if _f(r["fps"]) > 0] or [15.0])
    release_after = max_lost_frames / max(fps, 1.0) + 1.5
    bad = []
    for x in runs:
        start = _f(x[0]["t"])
        late = [r for r in x if _f(r["t"]) >= start + decay_s]
        if any(_cmd_mag(r) > 0.10 for r in late):
            bad.append(f"t={start:.1f}s: commands did not decay within {decay_s:.0f}s")
        dur = _f(x[-1]["t"]) - start
        if dur > release_after and all(r["locked_id"] != "" for r in x):
            bad.append(f"t={start:.1f}s: lock not released after {dur:.1f}s "
                       f"(expected within {release_after:.1f}s)")
    return (FAIL, "; ".join(bad)) if bad else (PASS, f"{len(runs)} loss event(s) handled")


def check_failsafes(rows):
    fs = [r for r in rows if r["safety_action"] not in ("", "NONE")]
    if not fs:
        return INCONCLUSIVE, "no failsafe ever triggered — kill the tracker/video/link mid-flight"
    bad = [r for r in fs if _cmd_mag(r) > 1e-6]
    if bad:
        return FAIL, f"{len(bad)} frame(s) sent motion commands while a failsafe was active (t={_f(bad[0]['t']):.1f}s)"
    po = [i for i, r in enumerate(rows) if r["safety_reason"].startswith("pilot_override")]
    if po and rows[po[0]]["locked_id"] != "" and \
            any(r["locked_id"] != "" for r in rows[po[0] + 2:po[0] + 12]):
        return FAIL, "lock persisted after pilot override"
    kinds = sorted({r["safety_action"] for r in fs})
    return PASS, f"actions seen: {', '.join(kinds)}; commands suppressed in all {len(fs)} frames"


def check_soak(rows, min_seconds, max_errors=5, min_fps=8.0):
    dur = _f(rows[-1]["t"])
    if dur < min_seconds:
        return INCONCLUSIVE, f"run lasted {dur:.0f}s (need {min_seconds:.0f}s)"
    final_err = _f(rows[-1]["errors"])
    half = rows[len(rows) // 2:]
    growth = _f(half[-1]["errors"]) - _f(half[0]["errors"])
    fps = sorted(_f(r["fps"]) for r in rows if _f(r["fps"]) > 0)
    p5 = fps[int(len(fps) * 0.05)] if fps else 0.0
    ok = final_err <= max_errors and growth <= 1 and p5 >= min_fps
    return (PASS if ok else FAIL), \
        f"errors={final_err:.0f} (growth in 2nd half {growth:.0f}), fps p5={p5:.1f}"


# --------------------------------------------------------------------------
def run_all(rows, follow_seconds=60.0, soak_seconds=0.0, max_lost_frames=60, rest_window=None):
    results = [("log integrity", *check_log_integrity(rows))]
    if results[0][1] == FAIL:
        return results
    results += [
        ("follow target", *check_follow(rows, follow_seconds)),
        ("no creep at rest", *check_no_creep(rows, rest_window)),
        ("lock loss handling", *check_lock_loss(rows, max_lost_frames)),
        ("failsafes / override", *check_failsafes(rows)),
    ]
    if soak_seconds > 0:
        results.append(("soak", *check_soak(rows, soak_seconds)))
    return results


def lock_segments(rows):
    """[(locked_id, t_start, t_end, n_frames)] for contiguous airborne runs on one locked id."""
    segs, cur = [], None
    for r in rows:
        if r["airborne"] not in ("1", "True", "true") or r["locked_id"] == "":
            cur = None
            continue
        if cur and cur[0] == r["locked_id"]:
            cur[2], cur[3] = _f(r["t"]), cur[3] + 1
        else:
            cur = [r["locked_id"], _f(r["t"]), _f(r["t"]), 1]
            segs.append(cur)
    return [tuple(x) for x in segs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--follow-seconds", type=float, default=60.0)
    ap.add_argument("--soak-seconds", type=float, default=0.0)
    ap.add_argument("--max-lost-frames", type=int, default=60)
    ap.add_argument("--locks", action="store_true", help="list lock segments (id, start s, end s) to help pick --rest-window")
    ap.add_argument("--rest-window", type=float, nargs=2, metavar=("START", "END"),
                    help="seconds (log time) during which the locked target was stationary; enables the creep check")
    a = ap.parse_args(argv)
    rows_ = load(a.csv)
    if a.locks:
        print("lock segments (id, start_s, end_s, frames):")
        for sg in lock_segments(rows_):
            print(f"  id={sg[0]:>5}  {sg[1]:7.1f} -> {sg[2]:7.1f}  ({sg[3]} frames)")
        print()
    results = run_all(rows_, a.follow_seconds, a.soak_seconds, a.max_lost_frames,
                  tuple(a.rest_window) if a.rest_window else None)
    width = max(len(n) for n, _, _ in results)
    for name, status, detail in results:
        print(f"{status:<13} {name:<{width}}  {detail}")
    failed = any(s == FAIL for _, s, _ in results)
    print("\nRESULT:", "FAIL" if failed else
          ("PASS" if all(s == PASS for _, s, _ in results) else "INCOMPLETE (some checks inconclusive)"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
