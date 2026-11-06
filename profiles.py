"""Mission profiles: named overrides merged onto config.yaml (--profile NAME).

A profile may override any key (usually flight.max_forward_speed and the pid block).
Unknown profile -> ValueError listing the available ones, so a typo cannot silently fly
with the wrong speed limits.
"""
import copy


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_profile(cfg: dict, name: str) -> dict:
    profiles = cfg.get('profiles', {}) or {}
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; available: {sorted(profiles) or 'none'}")
    merged = _merge({k: v for k, v in cfg.items() if k != 'profiles'}, profiles[name] or {})
    merged['active_profile'] = name
    return merged
