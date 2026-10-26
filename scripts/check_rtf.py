#!/usr/bin/env python3
"""How fast is the simulator running compared with real time?   (sim must be running)

    python3 scripts/check_rtf.py [--seconds 10]

RTF 1.0 = real time. Below ~0.9 the drone's world runs slower than the wall clock, so the tracker sees fewer
frames per simulated second and everything is harder; targets moved by wall-clock would be effectively too fast.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sim"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--world", default="tracker_world")
    a = ap.parse_args()
    from sim_clock import SimClock
    c = SimClock(a.world)
    if not c.ok:
        sys.exit("could not subscribe to the world stats topic: is the sim running?")
    time.sleep(a.seconds)
    r = c.rtf()
    if r is None:
        sys.exit("no world statistics received")
    verdict = "OK" if r >= 0.9 else "SLOW: lighten the sim load (see docs/DEMO.md) or the show uses sim time to compensate"
    print(f"real-time factor over {a.seconds:.0f} s: {r:.2f}   {verdict}")


if __name__ == "__main__":
    main()
