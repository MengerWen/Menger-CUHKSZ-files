"""Training protocol utilities using pandas, torch, and TensorBoard."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import random
import tempfile
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter


@dataclass(frozen=True)
class TrainingProtocol:
    """High-level out-of-sample training protocol."""

    scheme: str = "expanding"
    initial_train_months: int = 60
    validation_months: int = 12
    refit_frequency_months: int = 1
    seed: int = 20260408


def validate_training_protocol(protocol: TrainingProtocol) -> None:
    """Validate rolling/expanding protocol settings."""

    if protocol.scheme not in {"expanding", "rolling"}:
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    if protocol.initial_train_months <= 0:
        raise ValueError("initial_train_months must be positive")
    if protocol.validation_months <= 0:
        raise ValueError("validation_months must be positive")
    if protocol.refit_frequency_months <= 0:
        raise ValueError("refit_frequency_months must be positive")


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, numpy, and torch seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_torch_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA if available and requested, otherwise CPU."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_summary_writer(log_dir: Path | str) -> SummaryWriter:
    """Create a TensorBoard writer with an ASCII-path fallback.

    TensorBoard's Windows file writer can fail on paths containing non-ASCII or
    shell-special characters. If the project path is rejected, logs fall back to
    the user's home directory while preserving the requested relative suffix.
    """

    requested = Path(log_dir)
    project_root = Path(__file__).resolve().parents[2]
    path = requested if requested.is_absolute() else project_root / requested
    fallback_suffix = requested if not requested.is_absolute() else Path(requested.name)
    candidates = [path, Path.home() / "mfe5340_tensorboard_logs" / fallback_suffix, Path(tempfile.gettempdir()) / "mfe5340_tensorboard_logs" / fallback_suffix]
    errors: list[str] = []
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(candidate))
            if candidate != path:
                warnings.warn(f"TensorBoard logdir {path} failed; using {candidate}", RuntimeWarning)
            return writer
        except (OSError, PermissionError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise PermissionError("Unable to create TensorBoard SummaryWriter. Tried: " + " | ".join(errors))


def log_metrics(writer: SummaryWriter, metrics: pd.Series | dict[str, float], step: int, prefix: str = "metrics") -> None:
    """Log scalar metrics to TensorBoard from pandas or dict objects."""

    series = metrics if isinstance(metrics, pd.Series) else pd.Series(metrics, dtype="float64")
    for name, value in pd.to_numeric(series, errors="coerce").dropna().items():
        writer.add_scalar(f"{prefix}/{name}", float(value), global_step=step)


def make_oos_schedule(months: Iterable[pd.Timestamp] | pd.Series, protocol: TrainingProtocol) -> pd.DataFrame:
    """Create a pandas DataFrame describing OOS refit/evaluation months."""

    validate_training_protocol(protocol)
    month_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(months)).dropna().unique())).sort_values()
    rows: list[dict[str, pd.Timestamp]] = []
    start = protocol.initial_train_months + protocol.validation_months
    for pos in range(start, len(month_index), protocol.refit_frequency_months):
        train_start_pos = 0 if protocol.scheme == "expanding" else max(0, pos - protocol.initial_train_months - protocol.validation_months)
        rows.append(
            {
                "train_start": month_index[train_start_pos],
                "train_end": month_index[pos - protocol.validation_months - 1],
                "validation_start": month_index[pos - protocol.validation_months],
                "validation_end": month_index[pos - 1],
                "test_month": month_index[pos],
            }
        )
    return pd.DataFrame(rows)


def run_training(*_args: object, protocol: TrainingProtocol | None = None, **_kwargs: object) -> object:
    """Validate training settings; actual model fitting is reserved for modeling."""

    protocol = protocol or TrainingProtocol()
    validate_training_protocol(protocol)
    set_global_seed(protocol.seed)
    raise NotImplementedError("Actual model training is reserved for the modeling stage.")



