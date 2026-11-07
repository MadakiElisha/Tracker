"""csi_camera.CsiCamera against a fake Picamera2 (the real one only exists on a Pi)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csi_camera import CsiCamera, parse_csi_spec   # noqa: E402


class FakePicam:
    instances = []

    def __init__(self, index, autofocus=True, fail_after=None, channels=3):
        self.index = index
        self.camera_controls = {"FrameRate": (1, 120), "ExposureTime": (1, 100000)}
        if autofocus:
            self.camera_controls.update({"AfMode": (0, 2), "LensPosition": (0.0, 15.0)})
        self.controls_set = []
        self.config = None
        self.started = self.closed = False
        self.n = 0
        self.fail_after = fail_after
        self.channels = channels
        FakePicam.instances.append(self)

    def create_video_configuration(self, main, buffer_count, controls, sensor=None):
        return {"main": main, "buffer_count": buffer_count, "controls": controls, "sensor": sensor}

    def configure(self, cfg):
        self.config = cfg

    def set_controls(self, c):
        self.controls_set.append(dict(c))

    def start(self):
        self.started = True

    def capture_array(self, _name):
        time.sleep(0.005)
        self.n += 1
        if self.fail_after is not None and self.n > self.fail_after:
            raise RuntimeError("camera gone")
        w, h = self.config["main"]["size"]
        return np.full((h, w, self.channels), self.n % 255, dtype=np.uint8)

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def make(**kw):
    def factory(index):
        return FakePicam(index, **kw)
    return factory


def test_parse_spec():
    assert parse_csi_spec("csi") == 0
    assert parse_csi_spec("csi:") == 0
    assert parse_csi_spec("csi:1") == 1


def test_reads_bgr_frames_at_requested_size():
    cam = CsiCamera(0, 640, 480, 30, {}, picamera_cls=make())
    try:
        ok, f = cam.read()
        assert ok and f.shape == (480, 640, 3) and f.dtype == np.uint8
        fake = FakePicam.instances[-1]
        assert fake.config["main"] == {"size": (640, 480), "format": "RGB888"}
        assert fake.config["controls"]["FrameRate"] == 30.0
        assert cam.is_live and cam.isOpened() and cam.set(0, 0) is False
    finally:
        cam.release()
    assert FakePicam.instances[-1].closed


def test_autofocus_is_locked_at_infinity_by_default():
    cam = CsiCamera(0, 640, 480, 30, {}, picamera_cls=make(autofocus=True))
    cam.release()
    assert {"AfMode": 0, "LensPosition": 0.0} in FakePicam.instances[-1].controls_set


def test_fixed_focus_sensor_gets_no_af_controls():
    cam = CsiCamera(0, 640, 480, 30, {"lens_position": 2.0}, picamera_cls=make(autofocus=False))
    cam.release()
    assert all("AfMode" not in c for c in FakePicam.instances[-1].controls_set)


def test_short_shutter_applied_only_when_configured():
    cam = CsiCamera(0, 640, 480, 30, {"max_exposure_us": 3000}, picamera_cls=make())
    cam.release()
    assert {"ExposureTime": 3000} in FakePicam.instances[-1].controls_set
    cam = CsiCamera(0, 640, 480, 30, {"max_exposure_us": 0}, picamera_cls=make())
    cam.release()
    assert all("ExposureTime" not in c for c in FakePicam.instances[-1].controls_set)


def test_four_channel_frames_trimmed():
    cam = CsiCamera(0, 320, 240, 30, {}, picamera_cls=make(channels=4))
    try:
        ok, f = cam.read()
        assert ok and f.shape == (240, 320, 3)
    finally:
        cam.release()


def test_read_returns_newest_frame_not_a_backlog():
    cam = CsiCamera(0, 64, 48, 30, {}, picamera_cls=make())
    try:
        time.sleep(0.1)                       # ~20 frames captured meanwhile
        ok, f = cam.read()
        assert ok
        assert f[0, 0, 0] >= 10               # not frame 1: the slot holds the latest
        assert cam.frame_age() < 0.1
    finally:
        cam.release()


def test_camera_failure_reports_no_frame_instead_of_stale():
    cam = CsiCamera(0, 64, 48, 30, {}, picamera_cls=make(fail_after=2))
    try:
        time.sleep(0.15)
        cam.read(timeout=0.01)                # drain whatever arrived before the failure
        ok, frame = cam.read(timeout=0.1)
        # the camera died: no fresh frame, never the last good one -> tracker treats it as video loss
        assert ok is False and frame is None
        assert cam.frame_age() > 0.1
    finally:
        cam.release()


def test_missing_picamera2_gives_actionable_error():
    import csi_camera
    if csi_camera.Picamera2 is not None:
        pytest.skip("running on a Pi")
    with pytest.raises(ImportError, match="python3-picamera2"):
        CsiCamera(0)


# ---- lens motor focus (V4L2) and rotation ---------------------------------------------------
from types import SimpleNamespace   # noqa: E402

from csi_camera import set_vcm_focus   # noqa: E402


class FakeV4l2:
    """Answers v4l2-ctl like the Pi: only `lens_dev` has focus_absolute."""

    def __init__(self, lens_dev="/dev/v4l-subdev3", sticky=True):
        self.lens_dev, self.sticky, self.value, self.calls = lens_dev, sticky, 0, []

    def __call__(self, cmd):
        self.calls.append(cmd)
        dev = cmd[2]
        if "--list-ctrls" in cmd:
            out = ("focus_absolute 0x009a090a (int) : min=0 max=4095 value=%d" % self.value
                   if dev == self.lens_dev else "exposure 0x00980911 (int) : min=4")
            return SimpleNamespace(returncode=0, stdout=out)
        if "--set-ctrl" in cmd:
            if self.sticky:
                self.value = int(cmd[-1].split("=")[1])
            return SimpleNamespace(returncode=0, stdout="")
        if "--get-ctrl" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"focus_absolute: {self.value}")
        return SimpleNamespace(returncode=1, stdout="")


DEVS = ["/dev/v4l-subdev0", "/dev/v4l-subdev1", "/dev/v4l-subdev2", "/dev/v4l-subdev3"]


def test_vcm_focus_finds_lens_subdev_and_sets_value():
    fake = FakeV4l2()
    assert set_vcm_focus(850, runner=fake, devices=DEVS) == "/dev/v4l-subdev3"
    assert fake.value == 850
    assert ["v4l2-ctl", "-d", "/dev/v4l-subdev3", "--set-ctrl", "focus_absolute=850"] in fake.calls


def test_vcm_focus_none_when_no_lens_motor():
    assert set_vcm_focus(500, runner=FakeV4l2(lens_dev="/dev/none"), devices=DEVS) is None


def test_vcm_focus_raises_when_value_does_not_stick():
    with pytest.raises(RuntimeError, match="read back 0"):
        set_vcm_focus(500, runner=FakeV4l2(sticky=False), devices=DEVS)


def test_camera_applies_vcm_focus_and_fails_loudly_without_motor(monkeypatch):
    import csi_camera
    monkeypatch.setattr(csi_camera.glob, "glob", lambda _p: DEVS)
    fake = FakeV4l2()
    cam = CsiCamera(0, 64, 48, 30, {"vcm_focus": 1200}, picamera_cls=make(autofocus=False),
                    focus_runner=fake)
    cam.release()
    assert cam.focus_device == "/dev/v4l-subdev3" and fake.value == 1200
    with pytest.raises(RuntimeError, match="no lens motor"):
        CsiCamera(0, 64, 48, 30, {"vcm_focus": 1200}, picamera_cls=make(),
                  focus_runner=FakeV4l2(lens_dev="/dev/none"))


def test_no_vcm_focus_by_default():
    cam = CsiCamera(0, 64, 48, 30, {}, picamera_cls=make(),
                    focus_runner=lambda cmd: pytest.fail("v4l2-ctl must not run"))
    cam.release()
    assert cam.focus_device is None


class GradientPicam(FakePicam):
    def capture_array(self, _name):
        time.sleep(0.005)
        w, h = self.config["main"]["size"]
        f = np.zeros((h, w, 3), np.uint8)
        f[0, 0] = 255                          # marker in the top-left corner
        return f


def test_rotate_180_flips_both_axes():
    cam = CsiCamera(0, 64, 48, 30, {"rotate": 180}, picamera_cls=lambda i: GradientPicam(i))
    try:
        ok, f = cam.read()
        assert ok and f[-1, -1, 0] == 255 and f[0, 0, 0] == 0
    finally:
        cam.release()


def test_rotate_rejects_other_angles():
    with pytest.raises(ValueError, match="0 or 180"):
        CsiCamera(0, 64, 48, 30, {"rotate": 90}, picamera_cls=make())


def test_sensor_mode_forced_only_when_configured():
    cam = CsiCamera(0, 64, 48, 30, {"sensor_size": [2328, 1748]}, picamera_cls=make())
    cam.release()
    assert FakePicam.instances[-1].config["sensor"] == {"output_size": (2328, 1748)}
    cam = CsiCamera(0, 64, 48, 30, {}, picamera_cls=make())
    cam.release()
    assert FakePicam.instances[-1].config["sensor"] is None
