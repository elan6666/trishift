from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import yaml


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def merged_dict(base: dict, override: dict | None) -> dict:
    out = deepcopy(base)
    if override:
        deep_update(out, override)
    return out


def load_yaml_file(path: str | Path) -> dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise TypeError(f"YAML root must be a mapping: {p}")
    return obj


def dump_yaml(path: str | Path, obj: dict, *, allow_unicode: bool = False) -> None:
    Path(path).write_text(
        yaml.safe_dump(obj, sort_keys=False, allow_unicode=allow_unicode),
        encoding="utf-8",
    )

