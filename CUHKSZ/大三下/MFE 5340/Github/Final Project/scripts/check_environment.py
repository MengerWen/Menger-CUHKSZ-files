"""Check the project Anaconda environment and key package versions."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "scipy",
    "statsmodels",
    "linearmodels",
    "torch",
    "torchvision",
    "torchaudio",
    "torch_geometric",
    "tensorboard",
    "einops",
    "networkx",
    "yaml",
    "tables",
    "h5py",
    "pyarrow",
]


def main() -> None:
    records: list[dict[str, str]] = []
    for package in PACKAGES:
        try:
            module = importlib.import_module(package)
            records.append({"package": package, "status": "ok", "version": str(getattr(module, "__version__", "installed"))})
        except Exception as exc:
            records.append({"package": package, "status": "error", "version": repr(exc)})
    print(f"python_executable: {sys.executable}")
    print(pd.DataFrame.from_records(records).to_string(index=False))


if __name__ == "__main__":
    main()
