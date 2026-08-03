#!/usr/bin/env python3
"""
Standalone diagnostic — connects with raw pymavlink only, finds the
real autopilot component, requests RC_CHANNELS explicitly, and prints
every RC_CHANNELS message raw. No threads, no queues, no abstraction.

Run this, then in the MAVProxy STABILIZE> prompt type:
    rc 7 2000
You should see chan7_raw jump to 2000 within ~1 second below.
"""
import time
from pymavlink import mavutil

print("Connecting...")
master = mavutil.mavlink_connection('udpin:127.0.0.1:14550')

AUTOPILOT_COMPONENT = mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1
print(f"Looking for heartbeat from component {AUTOPILOT_COMPONENT}...")

target_system = None
target_component = None
deadline = time.time() + 15
while time.time() < deadline:
    msg = master.recv_match(type='HEARTBEAT', blocking=True, timeout=2.0)
    if msg is None:
        continue
    print(f"  saw heartbeat: system={msg.get_srcSystem()} "
          f"component={msg.get_srcComponent()}")
    if msg.get_srcComponent() == AUTOPILOT_COMPONENT:
        target_system    = msg.get_srcSystem()
        target_component = msg.get_srcComponent()
        break

if target_system is None:
    print("ERROR: never saw a heartbeat from component 1. Aborting.")
    exit(1)

print(f"\nLocked onto System={target_system} Component={target_component}\n")

print("Requesting RC_CHANNELS (msg 65) at 10Hz...")
master.mav.command_long_send(
    target_system, target_component,
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
    65, 100000, 0, 0, 0, 0, 0
)

ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3.0)
if ack is None:
    print("WARNING: no COMMAND_ACK received at all.\n")
else:
    result_name = mavutil.mavlink.enums['MAV_RESULT'][ack.result].name
    print(f"COMMAND_ACK: command={ack.command} result={result_name}\n")

print("Now watching for RC_CHANNELS messages for 30s.")
print("In MAVProxy, type:  rc 7 2000   then   rc 8 2000   then   rc 9 1500\n")

t_end = time.time() + 30
seen_any = False
while time.time() < t_end:
    msg = master.recv_match(type='RC_CHANNELS', blocking=True, timeout=1.0)
    if msg is None:
        print("  ...no RC_CHANNELS message in the last 1s", end='\r')
        continue
    seen_any = True
    c7 = getattr(msg, 'chan7_raw', None)
    c8 = getattr(msg, 'chan8_raw', None)
    c9 = getattr(msg, 'chan9_raw', None)
    print(f"  RC_CHANNELS  ch7={c7}  ch8={c8}  ch9={c9}                ")

print()
if not seen_any:
    print("RESULT: Never received a single RC_CHANNELS message.")
    print("  -> The stream request was not honoured. Check the COMMAND_ACK")
    print("     result above. If it says DENIED or UNSUPPORTED, the")
    print("     ArduCopter firmware build may not support this message ID")
    print("     via SET_MESSAGE_INTERVAL — file that as the next thing to fix.")
else:
    print("RESULT: RC_CHANNELS messages ARE arriving. The fix worked.")
    print("  -> If ch7/ch8/ch9 never changed despite typing rc commands,")
    print("     the issue is on the MAVProxy/SITL side, not our code.")
