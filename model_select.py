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


# ---- class mapping ------------------------------------------------------------------------------
# The tracker's class ids (config.yaml `classes`, profiles, setpoints) are VisDrone ids. A model
# trained on another dataset (stock COCO YOLO) is mapped onto them by class NAME, so the same
# config, RC filter groups and distance setpoints work with either model.
VISDRONE_IDS = {'pedestrian': 0, 'people': 1, 'bicycle': 2, 'car': 3, 'van': 4, 'truck': 5,
                'tricycle': 6, 'awning-tricycle': 7, 'bus': 8, 'motor': 9}
ALIASES = {'person': 'pedestrian', 'motorcycle': 'motor', 'motorbike': 'motor'}


class ClassMap:
    """model class id <-> tracker (VisDrone) class id."""

    def __init__(self, model_names):
        if not model_names:                 # unknown (stub models in tests): assume VisDrone ids
            model_names = {i: n for n, i in VISDRONE_IDS.items()}
        names = dict(model_names.items()) if hasattr(model_names, 'items') else dict(enumerate(model_names))
        self.to_internal = {}
        for mid, name in names.items():
            key = ALIASES.get(str(name).lower(), str(name).lower())
            if key in VISDRONE_IDS:
                self.to_internal[int(mid)] = VISDRONE_IDS[key]
        self.identity = all(self.to_internal.get(i) == i for i in range(10)) and len(names) == 10
        self.from_internal = {}
        for mid, iid in self.to_internal.items():
            self.from_internal.setdefault(iid, []).append(mid)

    def model_classes(self, internal_ids):
        """Tracker class ids -> the model's ids, for the `classes=` filter (unknown ids dropped)."""
        if self.identity:
            return list(internal_ids)
        out = []
        for i in internal_ids:
            out.extend(self.from_internal.get(int(i), []))
        return sorted(set(out))

    def internal(self, model_id: int) -> int:
        """Model class id -> tracker id (-1 if the tracker has no such class)."""
        return int(model_id) if self.identity else self.to_internal.get(int(model_id), -1)

    def describe(self) -> str:
        if self.identity:
            return "VisDrone classes (no mapping)"
        inv = {v: k for k, v in VISDRONE_IDS.items()}
        return ", ".join(f"{m}->{inv[i]}" for m, i in sorted(self.to_internal.items()))
