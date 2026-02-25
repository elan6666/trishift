from __future__ import annotations

from pathlib import Path
import subprocess


def safe_git_commit(repo_root: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(repo_root)),
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    s = out.decode("utf-8", "replace").strip()
    return s or None

