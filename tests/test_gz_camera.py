import sys, types
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gz_camera import image_msg_to_bgr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import add_camera_to_iris as cam


def msg(w, h, data, fmt=3):
    return types.SimpleNamespace(width=w, height=h, data=data, pixel_format_type=fmt)


def test_rgb_to_bgr():
    d = bytes([255, 0, 0] * 4)  # red RGB
    out = image_msg_to_bgr(msg(2, 2, d))
    assert out.shape == (2, 2, 3) and tuple(out[0, 0]) == (0, 0, 255)


def test_bgr_passthrough():
    out = image_msg_to_bgr(msg(1, 1, bytes([1, 2, 3]), fmt=8))
    assert tuple(out[0, 0]) == (1, 2, 3)


def test_gray_and_rgba():
    assert image_msg_to_bgr(msg(2, 1, bytes([5, 6]), fmt=1)).shape == (1, 2, 3)
    assert image_msg_to_bgr(msg(1, 1, bytes([9, 8, 7, 255]), fmt=4)).shape == (1, 1, 3)


def test_bad_sizes():
    with pytest.raises(ValueError):
        image_msg_to_bgr(msg(2, 2, bytes(5)))
    with pytest.raises(ValueError):
        image_msg_to_bgr(msg(0, 0, b""))


def test_patch_idempotent_and_revert(tmp_path):
    d = tmp_path / "models/iris_with_standoffs"; d.mkdir(parents=True)
    f = d / "model.sdf"
    orig = "<sdf>\n  <model name='x'>\n    <link name='base_link'/>\n  </model>\n</sdf>\n"
    f.write_text(orig)
    assert cam.main([str(tmp_path)]) == 0
    once = f.read_text()
    assert "tracker_camera" in once and "/drone/camera" in once
    assert cam.main([str(tmp_path)]) == 0
    assert f.read_text() == once and once.count("<sensor") == 1
    assert cam.main([str(tmp_path), "--revert"]) == 0
    assert f.read_text() == orig
    assert (d / "model.sdf.orig").read_text() == orig


def test_missing_dir(tmp_path):
    assert cam.main([str(tmp_path)]) == 2
