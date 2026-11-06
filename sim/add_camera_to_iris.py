#!/usr/bin/env python3
"""Add a forward/down-tilted camera to the ArduPilot Gazebo iris model.

Usage:  python3 sim/add_camera_to_iris.py ~/ardupilot_gazebo [--pitch-deg 20] [--revert]

Edits models/iris_with_standoffs/model.sdf: appends a link + fixed joint
carrying a camera sensor published on gz topic /drone/camera. Idempotent
(marker-guarded) and keeps a .orig backup. UNTESTED against a live Gazebo.
"""
import argparse, math, shutil, sys
from pathlib import Path

BEGIN, END = "<!-- TRACKER_CAMERA_BEGIN -->", "<!-- TRACKER_CAMERA_END -->"
TOPIC = "/drone/camera"


def snippet(pitch_deg: float) -> str:
    pitch = math.radians(pitch_deg)  # positive pitch = nose down in SDF
    return f"""    {BEGIN}
    <link name='tracker_camera_link'>
      <pose>0.12 0 0.02 0 {pitch:.4f} 0</pose>
      <inertial><mass>0.01</mass>
        <inertia><ixx>1e-6</ixx><iyy>1e-6</iyy><izz>1e-6</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <sensor name='tracker_camera' type='camera'>
        <camera>
          <horizontal_fov>1.5708</horizontal_fov>
          <image><width>640</width><height>480</height><format>R8G8B8</format></image>
          <clip><near>0.1</near><far>200</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
        <topic>{TOPIC}</topic>
      </sensor>
    </link>
    <joint name='tracker_camera_joint' type='fixed'>
      <parent>base_link</parent>
      <child>tracker_camera_link</child>
    </joint>
    {END}
"""


def patch(text: str, pitch_deg: float) -> str:
    if BEGIN in text:
        text = revert(text)
    idx = text.rfind("</model>")
    if idx < 0:
        raise ValueError("no </model> found")
    return text[:idx] + snippet(pitch_deg) + "  " + text[idx:]


def revert(text: str) -> str:
    if BEGIN not in text:
        return text
    a = text.index(BEGIN)
    a = text.rfind("\n", 0, a) + 1
    b = text.index(END) + len(END)
    b = text.find("\n", b) + 1
    return text[:a] + text[b:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ardupilot_gazebo")
    ap.add_argument("--pitch-deg", type=float, default=20.0)
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args(argv)
    f = Path(a.ardupilot_gazebo).expanduser() / "models/iris_with_standoffs/model.sdf"
    if not f.is_file():
        print(f"not found: {f}", file=sys.stderr)
        return 2
    bak = f.with_suffix(".sdf.orig")
    if not bak.exists():
        shutil.copy2(f, bak)
    t = f.read_text()
    f.write_text(revert(t) if a.revert else patch(t, a.pitch_deg))
    print(("reverted " if a.revert else "patched ") + str(f))
    return 0


if __name__ == "__main__":
    sys.exit(main())
