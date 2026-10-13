"""Pandas-based feature manifest helpers for characteristic folders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loaders import list_feature_files


@dataclass(frozen=True)
class FeatureManifest:
    """A feature manifest for one universe."""

    universe: str
    files: tuple[Path, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(path.stem for path in self.files)

    def to_frame(self) -> pd.DataFrame:
        """Return the manifest as a pandas DataFrame."""

        return pd.DataFrame(
            {
                "universe": self.universe,
                "feature": [path.stem for path in self.files],
                "path": [str(path) for path in self.files],
                "bytes": [path.stat().st_size for path in self.files],
            }
        )


def build_feature_manifest(universe: str = "features500", root: Path | None = None) -> FeatureManifest:
    """Create a pandas-exportable feature manifest without loading values."""

    return FeatureManifest(universe=universe, files=tuple(list_feature_files(universe, root=root)))
