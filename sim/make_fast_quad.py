#!/usr/bin/env python3
"""Make an OPT-IN stronger-thrust copy of the ArduPilot Gazebo Iris, plus a matching highway world.

    python3 sim/make_fast_quad.py ~/ardupilot_gazebo [--thrust-scale 1.4] [--camera-pitch-deg 20]

Why: the stock Iris is the right size class (1.5 kg, ~10 inch props) but its thrust-to-weight is ~2; a
10-15 inch racing/cinewhoop-class quad is 4+, so it can lean far harder and accelerate faster. The copy
multiplies the rotor speed command by --thrust-scale (thrust ~ scale^2). Nothing stock is modified:
it creates models/fast_quad and models/fast_quad_with_ardupilot, and sim/worlds/highway_world_fast.sdf.

Validate in this order (EXPERIMENTAL, not yet run in Gazebo):
  1. --thrust-scale 1.0   must hover and fly exactly like the stock Iris (proves the copy is sound)
  2. --thrust-scale 1.4   hover throttle drops; ArduPilot relearns it (MOT_HOVER_LEARN), so re-run takeoff twice
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OLD_BASE, OLD_TOP = "iris_with_standoffs", "iris_with_ardupilot"
NEW_BASE, NEW_TOP = "fast_quad", "fast_quad_with_ardupilot"


def rename(text: str) -> str:
    # longer name first: 'iris_with_ardupilot' does not contain 'iris_with_standoffs', but be explicit
    return text.replace(OLD_TOP, NEW_TOP).replace(OLD_BASE, NEW_BASE)


def scale_motors(text: str, scale: float) -> str:
    """Scale each rotor's speed command by `scale` and its torque limit by scale^2 (drag torque ~ w^2)."""
    n_mult = len(re.findall(r"<multiplier>\s*-?[0-9.]+\s*</multiplier>", text))
    if n_mult < 4:
        raise ValueError(f"expected >= 4 <multiplier> entries, found {n_mult}: is this the ArduPilot Iris model?")

    def mult(m):
        v = float(m.group(1)) * scale
        return f"<multiplier>{v:g}</multiplier>"

    def cmd(m):
        v = float(m.group(2)) * scale * scale
        return f"<{m.group(1)}>{v:g}</{m.group(1)}>"

    text = re.sub(r"<multiplier>\s*(-?[0-9.]+)\s*</multiplier>", mult, text)
    return re.sub(r"<(cmd_max|cmd_min)>\s*(-?[0-9.]+)\s*</\1>", cmd, text)


def make(models_dir: str, scale: float, camera_pitch_deg: float):
    src_base, src_top = os.path.join(models_dir, OLD_BASE), os.path.join(models_dir, OLD_TOP)
    for d in (src_base, src_top):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} not found (is {models_dir} the ardupilot_gazebo/models folder?)")
    dst_base, dst_top = os.path.join(models_dir, NEW_BASE), os.path.join(models_dir, NEW_TOP)
    for s, d in ((src_base, dst_base), (src_top, dst_top)):
        if os.path.isdir(d):
            shutil.rmtree(d)
        shutil.copytree(s, d)
    for d in (dst_base, dst_top):
        for name in ("model.sdf", "model.config"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                with open(p) as f:
                    t = rename(f.read())
                if d == dst_top and name == "model.sdf":
                    t = scale_motors(t, scale)
                with open(p, "w") as f:
                    f.write(t)
    # camera on the copy, same mount as the stock Iris camera
    from add_camera_to_iris import patch
    p = os.path.join(dst_base, "model.sdf")
    with open(p) as f:
        t = f.read()
    with open(p, "w") as f:
        f.write(patch(t, camera_pitch_deg))
    from make_highway_world import build
    out = os.path.join(HERE, "worlds", "highway_world_fast.sdf")
    with open(out, "w") as f:
        f.write(build(NEW_TOP))
    return dst_base, dst_top, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ardupilot_gazebo", help="path to ardupilot_gazebo (contains models/)")
    ap.add_argument("--thrust-scale", type=float, default=1.4, help="rotor speed multiplier (default 1.4 = ~2x thrust)")
    ap.add_argument("--camera-pitch-deg", type=float, default=20.0)
    a = ap.parse_args()
    if not 0.5 <= a.thrust_scale <= 2.0:
        sys.exit("--thrust-scale must be between 0.5 and 2.0")
    try:
        base, top, world = make(os.path.join(os.path.expanduser(a.ardupilot_gazebo), "models"),
                                a.thrust_scale, a.camera_pitch_deg)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"[ERROR] {e}")
    print(f"created {base}\ncreated {top}  (rotor speed x{a.thrust_scale}, torque limit x{a.thrust_scale ** 2:.2f})\nwrote {world}")
    print("run:  sim/launch_sim.sh ardupilot highway-fast")


if __name__ == "__main__":
    main()
