from __future__ import annotations

from pathlib import Path
import sys


def repo_root_from_file(file_path: str | Path, levels_up: int) -> Path:
    return Path(file_path).resolve().parents[int(levels_up)]


def ensure_on_sys_path(path: str | Path) -> None:
    s = str(Path(path))
    if s not in sys.path:
        sys.path.insert(0, s)


def ensure_repo_src_on_path(file_path: str | Path, levels_up: int) -> Path:
    repo_root = repo_root_from_file(file_path, levels_up)
    ensure_on_sys_path(repo_root / "src")
    return repo_root

