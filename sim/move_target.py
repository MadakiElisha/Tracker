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
    print("[TARGET MOVER] Person  → circular path, 20m radius, 40s per lap")
    print("[TARGET MOVER] Vehicle → back-and-forth patrol, X axis")
    print("[TARGET MOVER] Ctrl+C to stop.\n")

    t = 0.0
    INTERVAL = 0.1  # 10Hz is now perfectly safe because the node stays open

    try:
        while True:
            # --- Person target: circular path ---
            angle = (2 * math.pi * t) / 40.0
            px = 20.0 * math.cos(angle)
            py = 20.0 * math.sin(angle)
            yaw_p = angle + math.pi / 2
            
            # Construct the Pose Protobuf message
            req_p = Pose()
            req_p.name = "moving_target"
            req_p.position.x = px
            req_p.position.y = py
            req_p.position.z = 0.9
            req_p.orientation.w = math.cos(yaw_p / 2.0)
            req_p.orientation.z = math.sin(yaw_p / 2.0)
            
            # Execute service call (timeout 500ms)
            node.request(service_name, req_p, Pose, Boolean, 500)

            # --- Vehicle target: sinusoidal patrol ---
            vx = 20.0 * math.sin((2 * math.pi * t) / 60.0)
            
            req_v = Pose()
            req_v.name = "vehicle_target"
            req_v.position.x = vx
            req_v.position.y = -15.0
            req_v.position.z = 0.75
            req_v.orientation.w = 1.0  # Zero rotation
            
            node.request(service_name, req_v, Pose, Boolean, 500)

            t += INTERVAL
            time.sleep(INTERVAL)

    except KeyboardInterrupt:
        print("\n[TARGET MOVER] Stopped cleanly.")

if __name__ == '__main__':
    main()
