#!/bin/bash
# =============================================================
# Drone Tracker — Full Simulation Launcher
# Usage: ./launch_sim.sh [ardupilot|px4] [scene]
#   scene (optional, default = the original small world, unchanged):
#     highway       2 km road + fast SITL limits (stock Iris)
#     highway-fast  same, with the stronger-thrust quad made by sim/make_fast_quad.py
# =============================================================

FC=${1:-ardupilot}
SCENE=${2:-default}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

WORLD_FILE="$SCRIPT_DIR/worlds/tracker_world.sdf"
PARAM_FILE="$SCRIPT_DIR/params/ardupilot_tracker.param"
case "$SCENE" in
    default) ;;
    highway|highway-fast)
        WORLD_FILE="$SCRIPT_DIR/worlds/highway_world.sdf"
        if [ "$SCENE" = "highway-fast" ]; then
            WORLD_FILE="$SCRIPT_DIR/worlds/highway_world_fast.sdf"
        fi
        if [ ! -f "$WORLD_FILE" ]; then
            echo "[ERROR] $WORLD_FILE not found."
            echo "  Generate it: python3 $SCRIPT_DIR/make_highway_world.py" \
                 "$([ "$SCENE" = highway-fast ] && echo '--drone fast_quad_with_ardupilot --out '"$WORLD_FILE")"
            exit 1
        fi
        # tracker params first, fast limits after (later lines win)
        PARAM_FILE="$(mktemp /tmp/ardupilot_merged_XXXX.param)"
        cat "$SCRIPT_DIR/params/ardupilot_tracker.param" "$SCRIPT_DIR/params/ardupilot_fast.param" > "$PARAM_FILE"
        ;;
    *) echo "[ERROR] unknown scene '$SCENE' (use: default | highway | highway-fast)"; exit 1 ;;
esac

echo "=============================================="
echo " Drone Tracker Simulation — FC: $FC, scene: $SCENE"
echo "=============================================="

cleanup() {
    echo ""
    echo "[SHUTDOWN] Stopping all processes..."
    [ -n "$SITL_PID" ] && kill $SITL_PID 2>/dev/null
    [ -n "$GZ_PID"   ] && kill $GZ_PID   2>/dev/null
    sleep 1
    echo "[SHUTDOWN] Done."
}
trap cleanup SIGINT SIGTERM EXIT

if [ "$FC" = "ardupilot" ]; then

    # ----------------------------------------------------------
    # CHNAGE 1: Enable Monitor Mode (Job Control)
    # This allows us to use 'fg' to bring the backgrounded SITL
    # process back to the foreground later.
    # ----------------------------------------------------------
    set -m

    # ----------------------------------------------------------
    # STAGE 1: Start Gazebo first.
    # ----------------------------------------------------------
    echo ""
    echo "[STAGE 1/3] Starting Gazebo Harmonic..."

    export GZ_SIM_RESOURCE_PATH="$HOME/ardupilot_gazebo/models:$HOME/ardupilot_gazebo/worlds:$GZ_SIM_RESOURCE_PATH"
    export GZ_SIM_SYSTEM_PLUGIN_PATH="$HOME/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH"

    gz sim -r "$WORLD_FILE" &
    GZ_PID=$!

    # ----------------------------------------------------------
    # STAGE 2: Wait for Gazebo simulation to actually be running.
    # ----------------------------------------------------------
    echo "[STAGE 2/3] Waiting for Gazebo world to be live..."
    echo "  (querying /world/tracker_world/stats topic)"
    WAIT=0
    while true; do
        STATS=$(timeout 2 gz topic -e -t /world/tracker_world/stats 2>/dev/null)
        if echo "$STATS" | grep -q "sim_time"; then
            echo "  Gazebo world confirmed live after ${WAIT}s."
            break
        fi

        sleep 1
        WAIT=$((WAIT + 1))
        echo "  ...waiting ${WAIT}s"

        if ! kill -0 $GZ_PID 2>/dev/null; then
            echo ""
            echo "[ERROR] Gazebo process died unexpectedly."
            echo "  Check the SDF file for errors."
            exit 1
        fi

        if [ $WAIT -ge 180 ]; then
            echo ""
            echo "[ERROR] Gazebo world did not start after 180s (first run downloads models: run sim/fetch_models.sh)."
            cleanup
            exit 1
        fi
    done

    echo "  Giving ArduPilotPlugin 2s to bind physics socket..."
    sleep 2

    # ----------------------------------------------------------
    # STAGE 3: Now start SITL.
    # ----------------------------------------------------------
    echo "[STAGE 3/3] Starting ArduPilot SITL..."
    export PATH="$PATH:$HOME/ardupilot/Tools/autotest"
    if ! command -v sim_vehicle.py >/dev/null 2>&1; then
        echo "[ERROR] sim_vehicle.py not found. Is ArduPilot cloned at ~/ardupilot?"
        cleanup
        exit 1
    fi
    cd ~/ardupilot
    
    sim_vehicle.py \
        -v ArduCopter \
        -f gazebo-iris \
        --model JSON \
        --add-param-file="$PARAM_FILE" \
        --console \
        --no-rebuild \
        --out=udp:127.0.0.1:14550 \
        --out=udp:127.0.0.1:14551 &
    SITL_PID=$!

    # Wait for SITL MAVProxy port to confirm it started
    echo "  Waiting for SITL to confirm startup..."
    WAIT=0
    until nc -z 127.0.0.1 5760 2>/dev/null; do
        sleep 1
        WAIT=$((WAIT + 1))
        echo "  ...waiting ${WAIT}s for SITL port 5760"
        if [ $WAIT -ge 40 ]; then
            echo "[ERROR] SITL did not open port 5760 after 40s."
            cleanup
            exit 1
        fi
    done
    echo "  SITL confirmed after ${WAIT}s."

    # ----------------------------------------------------------
    # All up. Print connection info.
    # ----------------------------------------------------------
    echo ""
    echo "=============================================="
    echo " ALL SYSTEMS RUNNING"
    echo ""
    echo "  Gazebo PID : $GZ_PID"
    echo "  SITL PID   : $SITL_PID"
    echo "  WSL2 IP    : $(hostname -I | awk '{print $1}')"
    echo ""
    echo "  QGroundControl (Windows):"
    echo "    Connection type : UDP"
    echo "    Port            : 14550"
    echo "    (no IP needed — QGC auto-detects via broadcast)"
    echo ""
    echo "  Next terminals to open:"
    echo "    Terminal 2 → python3 ~/drone_tracker/move_target.py"
    echo "    Terminal 3 → run DroneKit connection test"
    echo "=============================================="
    echo "Passing terminal control to MAVProxy prompt below..."
    echo ""

    # ----------------------------------------------------------
    # CHANGE 2: Hand standard input over to MAVProxy instead of 'wait'
    # ----------------------------------------------------------
    if [ -n "$DEMO_NONINTERACTIVE" ]; then
        # demo/run_demo.py starts this script without a terminal: nothing to hand control to, just stay alive
        wait $SITL_PID
    else
        fg %"sim_vehicle.py"
    fi

fi
