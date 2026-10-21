#!/usr/bin/env python3
"""
Moves simulated targets natively using Gazebo Harmonic's Python bindings.
No subprocess spam, no socket exhaustion.
"""
import time
import math
import sys

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
    # 1. Initialize a SINGLE persistent transport node
    node = Node()
    service_name = "/world/tracker_world/set_pose"

    print("[TARGET MOVER] Starting native gz-transport node.")
    print("[TARGET MOVER] Person → circle r=20m; person2 → line; pickup + SUV → patrols")
    print("[TARGET MOVER] Ctrl+C to stop.\n")

    t = 0.0
    INTERVAL = 0.1  # 10Hz is safe because the node stays open

    # name -> function(t) returning (x, y, z, yaw).  Names must exist in tracker_world.sdf.
    # Yaw is the direction of travel; if a Fuel model faces the wrong way, add an offset here.
    def person_circle(t):
        a = 2 * math.pi * t / 40.0
        return 20 * math.cos(a), 20 * math.sin(a), 0.0, a + math.pi / 2

    def person_line(t):                      # walks back and forth along y=8
        ph = 2 * math.pi * t / 50.0
        x = 20 + 10 * math.sin(ph)
        return x, 8.0, 0.0, 0.0 if math.cos(ph) > 0 else math.pi

    def pickup_patrol(t):                    # drives along y=-15
        ph = 2 * math.pi * t / 60.0
        return 20 * math.sin(ph), -15.0, 0.0, 0.0 if math.cos(ph) > 0 else math.pi

    def suv_patrol(t):                       # drives along y=15, different phase
        ph = 2 * math.pi * t / 80.0 + 1.0
        return 15 * math.sin(ph), 15.0, 0.0, 0.0 if math.cos(ph) > 0 else math.pi

    actors = {
        "moving_target": person_circle,
        "moving_person2": person_line,
        "vehicle_target": pickup_patrol,
        "moving_suv": suv_patrol,
    }

    try:
        while True:
            for name, path in actors.items():
                x, y, z, yaw = path(t)
                req = Pose()
                req.name = name
                req.position.x, req.position.y, req.position.z = x, y, z
                req.orientation.w = math.cos(yaw / 2.0)
                req.orientation.z = math.sin(yaw / 2.0)
                node.request(service_name, req, Pose, Boolean, 500)
            t += INTERVAL
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n[TARGET MOVER] Stopped cleanly.")

if __name__ == '__main__':
    main()
