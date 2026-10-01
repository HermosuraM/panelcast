"""Locate the project root (repo checkout, Databricks bundle folder, or $PANELCAST_ROOT)."""

from __future__ import annotations

import os
from pathlib import Path

_override: Path | None = None


def set_project_root(path: str | Path) -> None:
    global _override
    _override = Path(path)


def project_root() -> Path | None:
    if _override is not None:
        return _override
    env = os.environ.get("PANELCAST_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "conf").is_dir():
            return parent
    cwd = Path.cwd()
    if (cwd / "conf").is_dir():
        return cwd
    return None


def require_root() -> Path:
    root = project_root()
    if root is None:
        raise RuntimeError("Cannot find the PanelCast project root; pass --root or set PANELCAST_ROOT.")
    return root
