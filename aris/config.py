"""Minimal composable config: YAML + dotted CLI overrides.

Deliberately not Hydra for Step 1 -- one fewer dependency, and the override
semantics we need are ~40 lines. Swap in Hydra at Step 6 (sweeps) if you want
multirun; the dict shape below is Hydra-compatible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(s: str) -> Any:
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"none", "null"}:
        return None
    return s


def load(name: str = "base", overrides: list[str] | None = None) -> dict:
    """Load conf/<name>.yaml, apply `key.sub=value` overrides."""
    path = CONF_DIR / f"{name}.yaml"
    cfg = yaml.safe_load(path.read_text()) or {}
    for inc in cfg.pop("_include", []):
        cfg = _deep_merge(yaml.safe_load((CONF_DIR / f"{inc}.yaml").read_text()) or {}, cfg)
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"bad override {ov!r}, expected key.sub=value")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def config_hash(cfg: dict) -> str:
    """Stable hash of a config -- goes in the ledger so every number is traceable."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
