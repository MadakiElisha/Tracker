#!/usr/bin/env python3
"""Move the highway targets (sprinter and/or car) along the 2 km road in sim/worlds/highway_world.sdf.

    python3 sim/move_highway.py --target car --car-speed 20
    python3 sim/move_highway.py --target runner --runner-speed 10
    python3 sim/move_highway.py --target both --start-delay 20      # time to take off and lock first

Wall-clock based (a slow Gazebo call cannot slow the targets). The world name stays `tracker_world`,
so launch_sim.sh and the tracker work unchanged. Targets stop at the end of the road (x = 2000 m).
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import highway_paths as hp                                           # noqa: E402

RUNNER_Y = 8.5       # on the shoulder, beside the car lanes (road is y -7..+7, shoulder to y 10)
CAR_Y = -1.75


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["car", "runner", "both"], default="both")
    ap.add_argument("--car-speed", type=float, default=20.0, help="m/s (default 20)")
    ap.add_argument("--runner-speed", type=float, default=10.0, help="m/s (default 10)")
    ap.add_argument("--car-accel", type=float, default=2.5, help="m/s^2 (default 2.5)")
    ap.add_argument("--runner-accel", type=float, default=3.0, help="m/s^2 (default 3.0)")
    ap.add_argument("--start-delay", type=float, default=20.0,
                    help="seconds the targets stand still before moving (default 20: take off and lock first)")
    ap.add_argument("--lane-changes", type=float, default=0.0, metavar="SECONDS",
                    help="car changes lane every SECONDS seconds (default 0 = never)")
    ap.add_argument("--rate", type=float, default=30.0, help="pose updates per second (default 30)")
    ap.add_argument("--car-yaw-offset", type=float, default=0.0, help="radians, if the car model faces the wrong way")
    ap.add_argument("--runner-yaw-offset", type=float, default=0.0, help="radians, same for the runner")
    args = ap.parse_args()

    try:
        from gz.msgs10.pose_pb2 import Pose
        from gz.msgs10.boolean_pb2 import Boolean
        from gz.transport13 import Node
    except ImportError:
        print("[ERROR] Gazebo Python bindings not found: sudo apt install python3-gz-transport13")
        return 1

    node = Node()
    service = "/world/tracker_world/set_pose"
    actors = []
    if args.target in ("car", "both"):
        actors.append(("highway_car", args.car_speed, args.car_accel, CAR_Y, args.car_yaw_offset, args.lane_changes))
    if args.target in ("runner", "both"):
        actors.append(("highway_runner", args.runner_speed, args.runner_accel, RUNNER_Y, args.runner_yaw_offset, 0.0))
    print(f"[HIGHWAY] {', '.join(f'{n} {v:.1f} m/s' for n, v, *_ in actors)}; start in {args.start_delay:.0f} s; Ctrl+C stops")

    interval = 1.0 / max(1.0, args.rate)
    t0 = time.monotonic()
    failures = 0
    try:
        while True:
            t = time.monotonic() - t0
            for name, v, a, y0, yaw_off, lane_every in actors:
                x, y, yaw = hp.pose(t, v, a, y0, delay=args.start_delay, lane_every_s=lane_every)
                yaw += yaw_off
                req = Pose()
                req.name = name
                req.position.x, req.position.y, req.position.z = x, y, 0.0
                req.orientation.w, req.orientation.z = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
                ok, _ = node.request(service, req, Pose, Boolean, 500)
                failures = 0 if ok else failures + 1
            if failures > 50:
                print("[HIGHWAY] set_pose keeps failing: is the highway world running?")
                return 1
            time.sleep(max(0.0, interval - ((time.monotonic() - t0) - t)))
    except KeyboardInterrupt:
        print("\n[HIGHWAY] Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
