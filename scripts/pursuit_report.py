#!/usr/bin/env python3
"""What did the drone actually do in a chase? Speed vs command, detector dropouts, oscillation, re-ID.

    python3 scripts/pursuit_report.py logs/runs/<file>.csv

Needs a run log made by this version (it has the gs / v_fwd / tilt_deg / det_ms / reid columns).
"""
import argparse
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analyze_run as ar


def _col(rows, k):
    return [ar._f(r.get(k)) for r in rows if r.get(k, "") != ""]


def dropout_bursts(rows):
    """Lengths (seconds) of runs where a lock existed but the target was not visible."""
    out, start, last_t = [], None, None
    for r in rows:
        locked = r.get("locked_id", "") != ""
        vis = r.get("target_visible") in ("1", "True", "true", 1, True)
        t = ar._f(r["t"])
        if locked and not vis:
            start = t if start is None else start
            last_t = t
        else:
            if start is not None:
                out.append(last_t - start)
            start = None
    if start is not None:
        out.append(last_t - start)
    return out


def reversals(values, min_swing=2.0, window=3):
    """Count direction changes of cmd_vx whose swing is >= min_swing m/s (rocking indicator)."""
    pts, count, last_ext, direction = values, 0, None, 0
    for v in pts:
        if last_ext is None:
            last_ext = v
            continue
        if direction >= 0 and v < last_ext - min_swing:
            count += direction == 1
            direction, last_ext = -1, v
        elif direction <= 0 and v > last_ext + min_swing:
            count += direction == -1
            direction, last_ext = 1, v
        elif (direction >= 0 and v > last_ext) or (direction <= 0 and v < last_ext):
            last_ext = v
    return count


def summarise(rows):
    air = [r for r in rows if str(r.get("airborne")) in ("1", "True", "true")]
    lines = []
    if not air:
        return ["no airborne rows"]
    span = ar._f(air[-1]["t"]) - ar._f(air[0]["t"])
    fps = _col(air, "fps")
    lines.append(f"airborne {span:.0f}s, fps median {median(fps):.0f} min {min(fps):.0f}" if fps else f"airborne {span:.0f}s")
    det = sorted(_col(air, "det_ms"))
    if det:
        lines.append(f"detector+track ms: median {median(det):.0f}, p95 {det[int(0.95 * (len(det) - 1))]:.0f}")
    gs, cmd = _col(air, "gs"), _col(air, "cmd_vx")
    vf = _col(air, "v_fwd")
    if gs:
        lines.append(f"ground speed: max {max(gs):.1f} m/s, 95th pct {sorted(gs)[int(0.95 * (len(gs) - 1))]:.1f}")
    if cmd:
        lines.append(f"commanded forward speed: max {max(cmd):.1f}, min {min(cmd):.1f} m/s")
    if vf and len(vf) == len(cmd):
        short = sum(c - v > 3.0 for c, v in zip(cmd, vf)) / len(cmd)
        lines.append(f"drone slower than commanded by >3 m/s: {short:.0%} of frames"
                     + ("   <-- the drone is not reaching the commanded speed (SITL limit?)" if short > 0.15 else ""))
    tilt = _col(air, "tilt_deg")
    if tilt:
        lines.append(f"lean: max {max(tilt):.0f} deg, min {min(tilt):.0f} deg")
    bursts = dropout_bursts(air)
    if bursts:
        lines.append(f"detector dropouts while locked: {len(bursts)}, longest {max(bursts):.2f}s, "
                     f">0.3s: {sum(b > 0.3 for b in bursts)}, >1s: {sum(b > 1.0 for b in bursts)}")
    else:
        lines.append("detector dropouts while locked: none")
    if cmd:
        n = reversals(cmd)
        per10 = n / max(span, 1) * 10
        lines.append(f"speed-command swings >2 m/s: {n} ({per10:.1f} per 10s)"
                     + ("   <-- rocking" if per10 > 3 else ""))
    re = [r for r in air if r.get("reid") == "relocked"]
    srch = sum(1 for r in air if r.get("reid") == "search")
    lines.append(f"re-ID: {len(re)} re-lock(s), {srch} frames spent searching")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    rows = ar.load(ap.parse_args().csv)
    print("\n".join(summarise(rows)))
    print("\nlocks (id, start s, end s, frames):")
    for seg in ar.lock_segments(rows):
        print("  ", seg)


if __name__ == "__main__":
    main()
