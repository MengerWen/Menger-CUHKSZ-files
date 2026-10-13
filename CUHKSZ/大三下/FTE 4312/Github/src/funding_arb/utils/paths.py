"""Repository path helpers for scripts and tests."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_path(*parts: str) -> Path:
    """Resolve a path from the repository root."""
    return REPO_ROOT.joinpath(*parts)


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path

