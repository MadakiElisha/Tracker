"""Model and camera selection for tracker.py.

resolve_model(model_cfg, override)
    Picks what to load with ultralytics.YOLO:
      * an explicit --model path wins;
      * backend 'hailo' -> model.hailo_path (a *_hailo_model/ folder made by
        deployment/export_hailo.py), which Ultralytics runs on the Hailo NPU;
      * backend 'pytorch' -> model.path (.pt, PC / Gazebo);
      * backend 'auto' (default) -> Hailo when a Hailo device is present and the
        export exists, otherwise the .pt file.
    Returns (path, backend_name); raises FileNotFoundError with a fix-it hint.

apply_camera_preset(cfg, name)
    Merges camera.sensors[name] onto camera: field of view, mount pitch and CSI
    settings must describe the real lens, or the tilt correction and the
    yaw/pitch loops are scaled wrongly.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

HAILO_DEVICE = "/dev/hailo0"


def hailo_present(device: str = HAILO_DEVICE) -> bool:
    return os.path.exists(device)


def _is_hailo_export(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and p.name.endswith("_hailo_model") and any(p.rglob("*.hef"))


def resolve_model(model_cfg: dict, override: str | None = None, *, hailo_check=hailo_present):
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"Model not found: {override}")
        return override, ("hailo" if _is_hailo_export(override) else "pytorch")

    backend = str(model_cfg.get("backend", "auto")).lower()
    pt_path = model_cfg.get("path", "visdrone_yolo11n.pt")
    hef_dir = model_cfg.get("hailo_path", "visdrone_yolo11n_hailo_model")

    if backend not in ("auto", "hailo", "pytorch"):
        raise FileNotFoundError(f"model.backend must be auto, hailo or pytorch (got {backend!r})")

    if backend in ("auto", "hailo"):
        if _is_hailo_export(hef_dir) and (backend == "hailo" or hailo_check()):
            return hef_dir, "hailo"
        if backend == "hailo":
            raise FileNotFoundError(
                f"Hailo model folder not found or has no .hef: {hef_dir} "
                "(export on the PC: python3 deployment/export_hailo.py, then copy the folder to the Pi)")

    if not Path(pt_path).exists():
        raise FileNotFoundError(f"Model file not found: {pt_path}  (run: python3 model_download.py)")
    return pt_path, "pytorch"


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_camera_preset(cfg: dict, name: str) -> dict:
    cam = cfg.get("camera", {}) or {}
    sensors = cam.get("sensors", {}) or {}
    if name not in sensors:
        raise ValueError(f"unknown camera preset {name!r}; available: {sorted(sensors) or 'none'}")
    out = copy.deepcopy(cfg)
    preset = {k: v for k, v in (sensors[name] or {}).items() if k != "note"}
    out["camera"] = _merge(cam, preset)
    out["camera"]["sensor"] = name
    return out
