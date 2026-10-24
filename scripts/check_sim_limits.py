#!/usr/bin/env python3
"""Read the speed/lean limits the running SITL actually has, and say what chase speed they allow.

    python3 scripts/check_sim_limits.py                 # while sim/launch_sim.sh is running
    python3 scripts/check_sim_limits.py --connect udpin:127.0.0.1:14551

Uses the 14551 MAVProxy output (the tracker uses 14550), so it can run alongside the tracker.
"""
import argparse
import sys
import time

WANTED = ["WPNAV_SPEED", "WPNAV_ACCEL", "ANGLE_MAX", "GUID_TIMEOUT", "FRAME_CLASS", "FRAME_TYPE"]


def assess(values: dict) -> list:
    """Pure: turn parameter values into human-readable findings (tested without a simulator)."""
    out = []
    spd = values.get("WPNAV_SPEED")
    if spd is None:
        out.append(("WARN", "WPNAV_SPEED not readable: cannot tell the guided-mode speed limit."))
    else:
        ms = spd / 100.0
        if ms < 9.0:
            out.append(("FAIL", f"WPNAV_SPEED = {spd:g} cm/s: guided-mode speed is capped near {ms:.0f} m/s. "
                                f"A 10-20 m/s chase is impossible. Start with: sim/launch_sim.sh ardupilot highway"))
        elif ms < 22.0:
            out.append(("WARN", f"WPNAV_SPEED = {spd:g} cm/s: limit {ms:.0f} m/s. Enough for a 10 m/s runner, "
                                f"not for a 20 m/s car."))
        else:
            out.append(("OK", f"WPNAV_SPEED = {spd:g} cm/s: limit {ms:.0f} m/s."))
    acc = values.get("WPNAV_ACCEL")
    if acc is not None:
        out.append(("OK" if acc <= 800 else "WARN",
                    f"WPNAV_ACCEL = {acc:g} cm/s^2 ({acc / 100:.1f} m/s^2)"
                    + ("" if acc <= 800 else ": high accelerations lean the camera too far; the model fails at 9 m/s^2")))
    ang = values.get("ANGLE_MAX")
    if ang is not None:
        deg = ang / 100.0
        out.append(("OK" if deg >= 45 else "WARN",
                    f"ANGLE_MAX = {ang:g} cdeg ({deg:.0f} deg lean)"
                    + ("" if deg >= 45 else ": lean limited; fast chases will saturate near 12 m/s")))
    return out


def read_params(connect: str, timeout: float = 20.0) -> dict:
    from pymavlink import mavutil
    m = mavutil.mavlink_connection(connect)
    if m.wait_heartbeat(timeout=timeout) is None:
        raise SystemExit(f"No heartbeat on {connect}. Is sim/launch_sim.sh running?")
    vals = {}
    for name in WANTED:
        m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < 3.0:
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=1.0)
            if msg is not None and msg.param_id.rstrip("\x00") == name:
                vals[name] = float(msg.param_value)
                break
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--connect", default="udpin:127.0.0.1:14551")
    a = ap.parse_args()
    vals = read_params(a.connect)
    for k in WANTED:
        print(f"  {k:14s} {vals.get(k, 'n/a')}")
    print()
    bad = 0
    for level, text in assess(vals):
        print(f"{level:5s} {text}")
        bad += level == "FAIL"
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
