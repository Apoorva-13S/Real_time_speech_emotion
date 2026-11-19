"""Configuration loading: YAML files in configs/ + environment overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "local_train.yaml"


def _coerce(value: str) -> Any:
    """YAML-ish scalar coercion for environment overrides."""
    low = value.strip().lower()
    if low in ("true", "yes", "1", "on"):
        return True
    if low in ("false", "no", "0", "off"):
        return False
    if low in ("none", "null", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_yaml(path: str | Path) -> dict:
    """Load YAML without requiring PyYAML for the common flat case."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    try:
        import yaml
    except ImportError:                     # minimal fallback parser
        data: dict[str, Any] = {}
        for raw in path.read_text().splitlines():
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip() or line.startswith("-") or ":" not in line:
                continue
            key, _, val = line.partition(":")
            data[key.strip()] = _coerce(val.strip().strip("'\""))
        return data
    return yaml.safe_load(path.read_text()) or {}


def deep_update(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, val in extra.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = val
    return out


def env_overrides(prefix: str = "SER_") -> dict:
    """SER_TRAIN__EPOCHS=5 -> {"train": {"epochs": 5}}"""
    out: dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        node = out
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = _coerce(val)
    return out


def resolve(path: str | Path | None = None) -> dict:
    """Load a config file and apply SER_* environment overrides."""
    cfg_path = Path(path) if path else Path(os.environ.get("SER_CONFIG",
                                                          DEFAULT_CONFIG))
    return deep_update(load_yaml(cfg_path), env_overrides())


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


@dataclass
class RunPaths:
    """Where a run writes its artefacts. Every step prints these paths."""
    root: Path
    data: Path
    splits: Path
    model: Path
    reports: Path
    figures: Path
    extra: dict = field(default_factory=dict)

    @classmethod
    def create(cls, root: str | Path) -> "RunPaths":
        root = Path(root)
        obj = cls(
            root=root,
            data=root / "data",
            splits=root / "data" / "splits",
            model=root / "models" / "best",
            reports=root / "reports",
            figures=root / "reports" / "figures",
        )
        for p in (obj.data, obj.splits, obj.model, obj.reports, obj.figures):
            p.mkdir(parents=True, exist_ok=True)
        return obj
