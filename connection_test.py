#!/usr/bin/env python3
import time
import argparse
from dronekit import connect, VehicleMode

def arm_and_takeoff(vehicle, target_altitude):
    """
    Arms vehicle and fly to a target altitude with timeout protections.
    """
    print("  Switching to GUIDED...")
    vehicle.mode = VehicleMode("GUIDED")
    
    # Wait for mode change with a 5-second timeout
    timeout = time.time() + 5
    while vehicle.mode.name != "GUIDED":
        if time.time() > timeout:
            raise Exception("Timeout: Could not switch to GUIDED mode.")
        time.sleep(0.3)

    print("  Waiting for vehicle to be armable...")
    # This prevents the arming loop from hanging if pre-arm checks fail
    timeout = time.time() + 30
    while not vehicle.is_armable:
        if time.time() > timeout:
            raise Exception("Timeout: Vehicle is not armable (Check EKF/GPS).")
        time.sleep(1)

    print("  Arming motors...")
    vehicle.armed = True
    
    timeout = time.time() + 10
    while not vehicle.armed:
        if time.time() > timeout:
            raise Exception("Timeout: Motors failed to arm.")
        time.sleep(0.5)

    print(f"  Armed. Taking off to {target_altitude}m...")
    vehicle.simple_takeoff(target_altitude)

    # Wait until the vehicle reaches a safe height
    while True:
        alt = vehicle.location.global_relative_frame.alt
        print(f"  Altitude: {alt:.1f}m", end='\r')
        if alt >= target_altitude * 0.95:
            print(f"\n  Reached target altitude of {alt:.1f}m")
            break
        time.sleep(0.3)

def main():
    # Setup argument parsing for flexible networking
    parser = argparse.ArgumentParser(description='SITL Connection Test')
    parser.add_argument('--connect', default='127.0.0.1:14550', help="Vehicle connection target string.")
    args = parser.parse_args()

    print(f"Connecting to SITL on: {args.connect}...")
    vehicle = None

    try:
        vehicle = connect(args.connect, wait_ready=True, timeout=60)
        print(f"  Connected. Mode: {vehicle.mode.name}, Alt: {vehicle.location.global_relative_frame.alt:.1f}m")

        # Clear previous flight state safely
        if vehicle.mode.name in ['RTL', 'LAND']:
            print("  Previous flight in progress, waiting for landing...")
            while vehicle.location.global_relative_frame.alt > 0.3:
                time.sleep(0.5)
            time.sleep(3) # Let motors disarm safely
            print("  Landed.")

        # Execute flight maneuvers
        arm_and_takeoff(vehicle, 10)
        print("  SITL PIPELINE CONFIRMED HEALTHY")

    except KeyboardInterrupt:
        print("\n[USER ABORT] Script interrupted via keyboard.")
    except Exception as e:
        print(f"\n[ERROR] Flight aborted: {e}")
    finally:
        # The 'finally' block ensures the connection ALWAYS closes, 
        # even if the script crashes or you hit Ctrl+C.
        if vehicle:
            print("  Triggering RTL and closing connection...")
            vehicle.mode = VehicleMode("RTL")
            vehicle.close()
            print("  Cleanup complete.")

if __name__ == '__main__':
    main()
