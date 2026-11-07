"""Model selection (PyTorch vs Hailo export) and camera presets."""
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_select import resolve_model, apply_camera_preset   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _files(tmp_path, pt=True, hef=True, dirname="visdrone_yolo11n_hailo_model"):
    pt_path = tmp_path / "visdrone_yolo11n.pt"
    if pt:
        pt_path.write_bytes(b"x")
    d = tmp_path / dirname
    if hef:
        d.mkdir()
        (d / "visdrone_yolo11n.hef").write_bytes(b"x")
    return {"path": str(pt_path), "hailo_path": str(d)}


def test_auto_uses_hailo_when_device_and_export_present(tmp_path):
    cfg = _files(tmp_path)
    path, backend = resolve_model(cfg, hailo_check=lambda: True)
    assert backend == "hailo" and path == cfg["hailo_path"]


def test_auto_falls_back_to_pt_without_device(tmp_path):
    cfg = _files(tmp_path)
    path, backend = resolve_model(cfg, hailo_check=lambda: False)
    assert backend == "pytorch" and path == cfg["path"]


def test_auto_falls_back_to_pt_without_export(tmp_path):
    cfg = _files(tmp_path, hef=False)
    assert resolve_model(cfg, hailo_check=lambda: True)[1] == "pytorch"


def test_hailo_forced_without_export_explains(tmp_path):
    cfg = dict(_files(tmp_path, hef=False), backend="hailo")
    with pytest.raises(FileNotFoundError, match="export_hailo"):
        resolve_model(cfg, hailo_check=lambda: True)


def test_folder_without_hef_is_not_a_hailo_model(tmp_path):
    cfg = _files(tmp_path, hef=False)
    Path(cfg["hailo_path"]).mkdir()
    assert resolve_model(cfg, hailo_check=lambda: True)[1] == "pytorch"


def test_missing_everything_points_to_download(tmp_path):
    cfg = _files(tmp_path, pt=False, hef=False)
    with pytest.raises(FileNotFoundError, match="model_download"):
        resolve_model(cfg, hailo_check=lambda: False)


def test_override_wins_and_is_classified(tmp_path):
    cfg = _files(tmp_path)
    assert resolve_model(cfg, cfg["hailo_path"], hailo_check=lambda: False) == (cfg["hailo_path"], "hailo")
    assert resolve_model(cfg, cfg["path"], hailo_check=lambda: True) == (cfg["path"], "pytorch")
    with pytest.raises(FileNotFoundError):
        resolve_model(cfg, str(tmp_path / "nope.pt"))


def test_bad_backend_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="backend"):
        resolve_model(dict(_files(tmp_path), backend="tensorrt"))


def test_shipped_config_defaults():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    assert cfg["model"]["backend"] == "auto"
    assert cfg["model"]["hailo_path"].endswith("_hailo_model")
    assert cfg["camera"]["sensor"] is None          # sim behaviour unchanged by default
    assert cfg["camera"]["hfov_deg"] == 90.0


@pytest.mark.parametrize("name,hfov", [("gazebo", 90.0), ("imx296_m12_96", 83.3),
                                       ("imx519_78", 66.3), ("imx477_12mm", 29.4)])
def test_camera_presets(name, hfov):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    out = apply_camera_preset(cfg, name)
    assert out["camera"]["hfov_deg"] == hfov
    assert out["camera"]["sensor"] == name
    for k, v in cfg["camera"]["csi"].items():                    # base CSI keys survive the merge
        assert k in out["camera"]["csi"]
    assert out["camera"]["csi"]["fps"] == cfg["camera"]["csi"]["fps"]
    assert cfg["camera"]["hfov_deg"] == 90.0                     # input not mutated


def test_unknown_camera_preset_lists_choices():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    with pytest.raises(ValueError, match="imx296_m12_96"):
        apply_camera_preset(cfg, "imx999")


def test_camera_preset_survives_profile():
    from profiles import apply_profile
    cfg = apply_camera_preset(yaml.safe_load(open(ROOT / "config.yaml")), "imx519_78")
    for prof in cfg["profiles"]:
        assert apply_profile(cfg, prof)["camera"]["hfov_deg"] == 66.3, prof


def test_imx519_preset_full_view_and_locked_focus():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    csi = apply_camera_preset(cfg, "imx519_78")["camera"]["csi"]
    assert csi["sensor_size"] == [2328, 1748] and csi["vcm_focus"] == 1640
    assert csi["max_exposure_us"] == cfg["camera"]["csi"]["max_exposure_us"]   # base keys kept
    assert cfg["camera"]["csi"]["vcm_focus"] is None                          # sim/base untouched
