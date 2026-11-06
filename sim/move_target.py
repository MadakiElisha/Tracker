#!/usr/bin/env python3
"""
Moves simulated targets natively using Gazebo Harmonic's Python bindings.
No subprocess spam, no socket exhaustion.
"""
import time
import math
import sys
import argparse

# Import Gazebo Harmonic Python bindings
try:
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.transport13 import Node
except ImportError:
    print("[ERROR] Gazebo Python bindings not found.")
    print("Make sure you have the python transport layer installed:")
    print("  sudo apt install python3-gz-transport13")
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Move the simulated targets at chosen speeds.")
    ap.add_argument("--person-speed", type=float, default=1.4,
                    help="person on the circle / line, m/s (walk 1.4, jog 3, sprint 7). Default 1.4")
    ap.add_argument("--vehicle-speed", type=float, default=2.1,
                    help="peak speed of the pickup and SUV patrols, m/s. Default 2.1")
    ap.add_argument("--rate", type=float, default=20.0, help="pose updates per second (default 20)")
    args = ap.parse_args()
    # 1. Initialize a SINGLE persistent transport node
    node = Node()
    service_name = "/world/tracker_world/set_pose"

    print("[TARGET MOVER] Starting native gz-transport node.")
    print("[TARGET MOVER] Person → circle r=20m; person2 → line; pickup + SUV → patrols")
    print("[TARGET MOVER] Ctrl+C to stop.\n")

    INTERVAL = 1.0 / max(1.0, args.rate)
    vp, vv = max(0.1, args.person_speed), max(0.1, args.vehicle_speed)
    print(f"[TARGET MOVER] person {vp:.1f} m/s, vehicles peak {vv:.1f} m/s, {args.rate:.0f} Hz")

    # name -> function(t) returning (x, y, z, yaw).  Names must exist in tracker_world.sdf.
    # Yaw is the direction of travel; if a Fuel model faces the wrong way, add an offset here.
    def person_circle(t):
        a = vp * t / 20.0                  # angular rate = v / r, so speed is exactly vp
        return 20 * math.cos(a), 20 * math.sin(a), 0.0, a + math.pi / 2

    def person_line(t):                      # walks back and forth along y=8
        ph = vp * t / 10.0                 # amplitude 10 m -> peak speed vp
        x = 20 + 10 * math.sin(ph)
        return x, 8.0, 0.0, 0.0 if math.cos(ph) > 0 else math.pi

    def pickup_patrol(t):                    # drives along y=-15
        ph = vv * t / 20.0                 # amplitude 20 m -> peak speed vv
        return 20 * math.sin(ph), -15.0, 0.0, 0.0 if math.cos(ph) > 0 else math.pi

    def suv_patrol(t):                       # drives along y=15, different phase
        ph = vv * t / 15.0 + 1.0           # amplitude 15 m -> peak speed vv
        return 15 * math.sin(ph), 15.0, 0.0, 0.0 if math.cos(ph) > 0 else math.pi

    actors = {
        "moving_target": person_circle,
        "moving_person2": person_line,
        "vehicle_target": pickup_patrol,
        "moving_suv": suv_patrol,
    }

    t0 = time.monotonic()
    try:
        while True:
            t = time.monotonic() - t0      # wall clock: slow service calls cannot slow the targets down
            for name, path in actors.items():
                x, y, z, yaw = path(t)
                req = Pose()
                req.name = name
                req.position.x, req.position.y, req.position.z = x, y, z
                req.orientation.w = math.cos(yaw / 2.0)
                req.orientation.z = math.sin(yaw / 2.0)
                node.request(service_name, req, Pose, Boolean, 500)
            time.sleep(max(0.0, INTERVAL - ((time.monotonic() - t0) - t)))

    except KeyboardInterrupt:
        print("\n[TARGET MOVER] Stopped cleanly.")

if __name__ == '__main__':
    main()
