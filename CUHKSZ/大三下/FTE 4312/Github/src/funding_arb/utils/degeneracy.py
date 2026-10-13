"""Helpers for surfacing degenerate experiments instead of silently masking them."""

from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_SPLITS: tuple[str, ...] = ("train", "validation", "test")
_REGRESSION_TARGET_PATTERN = re.compile(r"^target_future_net_return_bps_(?P<horizon>.+)$")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return _safe_float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass(frozen=True)
class DegenerateExperimentDiagnostics:
    """Structured diagnostics attached to fail-fast errors and warnings."""

    stage: str
    reason: str
    status: str = "degenerate"
    model_name: str | None = None
    strategy_name: str | None = None
    source: str | None = None
    target_horizon: str | None = None
    split: str | None = None
    selected_threshold: float | None = None
    candidate_threshold_count: int | None = None
    valid_candidate_count: int | None = None
    signal_count_by_split: dict[str, int] = field(default_factory=dict)
    tradeable_rate_by_split: dict[str, float | None] = field(default_factory=dict)
    profitable_rate_by_split: dict[str, float | None] = field(default_factory=dict)
    future_net_return_bps_by_split: dict[str, dict[str, float | None]] = field(default_factory=dict)
    fallback_used: bool = False
    fallback_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _to_jsonable(asdict(self))

    def message(self) -> str:
        return (
            f"Degenerate experiment detected at stage='{self.stage}': {self.reason}. "
            f"Diagnostics={json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True)}"
        )


class DegenerateExperimentError(RuntimeError):
    """Raised when validation can no longer support meaningful model selection."""

    def __init__(self, diagnostics: DegenerateExperimentDiagnostics) -> None:
        self.diagnostics = diagnostics
        super().__init__(diagnostics.message())


def warn_on_degenerate_experiment(
    diagnostics: DegenerateExperimentDiagnostics,
    *,
    category: type[Warning] = RuntimeWarning,
    stacklevel: int = 2,
) -> None:
    """Emit a warning that preserves the structured diagnostic payload in the message."""

    warnings.warn(diagnostics.message(), category, stacklevel=stacklevel)


def infer_horizon_label(column_name: str | None) -> str | None:
    """Infer a target horizon label like ``24h`` from a canonical regression column."""

    if column_name is None:
        return None
    match = _REGRESSION_TARGET_PATTERN.match(column_name)
    if match is None:
        return None
    return match.group("horizon")


def infer_tradeable_column(regression_column: str | None) -> str | None:
    horizon = infer_horizon_label(regression_column)
    if horizon is None:
        return None
    return f"target_is_tradeable_{horizon}"


def infer_profitable_column(regression_column: str | None) -> str | None:
    horizon = infer_horizon_label(regression_column)
    if horizon is None:
        return None
    return f"target_is_profitable_{horizon}"


def summarize_threshold_search(search_frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize threshold-search candidate quality in a report-friendly way."""

    if search_frame.empty:
        return {
            "candidate_count": 0,
            "valid_candidate_count": 0,
            "all_objective_values_nan": True,
            "selected_threshold": None,
            "selected_objective_value": None,
            "status": "no_threshold_candidates",
            "reason": "Threshold search produced no candidate rows.",
        }
    objective_values = pd.to_numeric(search_frame.get("objective_value"), errors="coerce")
    selected_mask = (
        search_frame["selected"].fillna(False).astype(bool)
        if "selected" in search_frame.columns
        else pd.Series(False, index=search_frame.index)
    )
    selected_rows = search_frame.loc[selected_mask]
    selected_threshold = (
        _safe_float(pd.to_numeric(selected_rows["threshold"], errors="coerce").iloc[0])
        if not selected_rows.empty and "threshold" in selected_rows.columns
        else None
    )
    selected_objective_value = (
        _safe_float(pd.to_numeric(selected_rows["objective_value"], errors="coerce").iloc[0])
        if not selected_rows.empty and "objective_value" in selected_rows.columns
        else None
    )
    valid_candidate_count = int(objective_values.notna().sum())
    status = "ok"
    reason = None
    if valid_candidate_count == 0:
        status = "no_valid_threshold_candidates"
        reason = "All threshold candidates have NaN or missing objective_value."
    return {
        "candidate_count": int(len(search_frame)),
        "valid_candidate_count": valid_candidate_count,
        "all_objective_values_nan": bool(valid_candidate_count == 0),
        "selected_threshold": selected_threshold,
        "selected_objective_value": selected_objective_value,
        "objective_value_min": _safe_float(objective_values.min()),
        "objective_value_max": _safe_float(objective_values.max()),
        "status": status,
        "reason": reason,
    }


def label_split_diagnostics(
    frame: pd.DataFrame,
    *,
    split_column: str,
    net_return_column: str,
    tradeable_column: str | None,
    profitable_column: str | None,
    tradeable_threshold_bps: float,
    profitable_threshold_bps: float,
    split_names: Iterable[str] = DEFAULT_SPLITS,
) -> dict[str, dict[str, Any]]:
    """Compute split-aware label diagnostics used by training and reporting layers."""

    summaries: dict[str, dict[str, Any]] = {}
    for split_name in split_names:
        subset = frame.loc[frame[split_column].astype(str) == split_name].copy()
        net_returns = pd.to_numeric(subset.get(net_return_column), errors="coerce").dropna()
        tradeable = (
            pd.to_numeric(subset.get(tradeable_column), errors="coerce").dropna()
            if tradeable_column and tradeable_column in subset.columns
            else pd.Series(dtype=float)
        )
        profitable = (
            pd.to_numeric(subset.get(profitable_column), errors="coerce").dropna()
            if profitable_column and profitable_column in subset.columns
            else pd.Series(dtype=float)
        )
        tradeable_rate = _safe_float(tradeable.mean()) if not tradeable.empty else None
        profitable_rate = _safe_float(profitable.mean()) if not profitable.empty else None
        tradeable_positive_count = int((net_returns > float(tradeable_threshold_bps)).sum())
        profitable_positive_count = int((net_returns > float(profitable_threshold_bps)).sum())
        reason_parts: list[str] = []
        status = "ok"
        if subset.empty or net_returns.empty:
            status = "no_valid_labels"
            reason_parts.append(
                f"No valid '{net_return_column}' labels are available for split '{split_name}'."
            )
        else:
            max_net_return = _safe_float(net_returns.max())
            if tradeable_rate == 0.0:
                status = "no_tradeable_positive_labels"
                reason_parts.append("tradeable_rate == 0")
            if profitable_rate == 0.0:
                if status == "ok":
                    status = "no_profitable_positive_labels"
                reason_parts.append("profitable_rate == 0")
            if max_net_return is not None and max_net_return <= float(tradeable_threshold_bps):
                if tradeable_positive_count == 0:
                    if status == "ok":
                        status = "returns_never_clear_trade_threshold"
                    reason_parts.append(
                        "future_net_return_bps never exceeds the trade threshold on this split"
                    )
        summaries[split_name] = {
            "row_count": int(len(subset)),
            "valid_row_count": int(net_returns.shape[0]),
            "tradeable_rate": tradeable_rate,
            "profitable_rate": profitable_rate,
            "tradeable_positive_count": int(tradeable_positive_count),
            "profitable_positive_count": int(profitable_positive_count),
            "future_net_return_bps": {
                "min": _safe_float(net_returns.min()),
                "max": _safe_float(net_returns.max()),
                "mean": _safe_float(net_returns.mean()),
            },
            "status": status,
            "reason": "; ".join(reason_parts) if reason_parts else None,
            "supports_threshold_selection": status == "ok",
        }
    return summaries


def signal_split_diagnostics(
    frame: pd.DataFrame,
    *,
    split_column: str = "split",
    signal_column: str = "should_trade",
    split_names: Iterable[str] = DEFAULT_SPLITS,
) -> dict[str, dict[str, Any]]:
    """Compute split-aware signal diagnostics for manifests and warnings."""

    summaries: dict[str, dict[str, Any]] = {}
    for split_name in split_names:
        subset = frame.loc[frame[split_column].astype(str) == split_name].copy()
        signal_values = pd.to_numeric(subset.get(signal_column), errors="coerce").fillna(0.0)
        signal_count = int(signal_values.sum()) if not subset.empty else 0
        status = "ok"
        reason = None
        if subset.empty:
            status = "missing_split"
            reason = f"No rows are available for split '{split_name}'."
        elif signal_count == 0:
            status = "no_tradable_signals"
            reason = f"signal_count == 0 for split '{split_name}'."
        summaries[split_name] = {
            "row_count": int(len(subset)),
            "signal_count": signal_count,
            "signal_rate": _safe_float(signal_values.mean()) if not subset.empty else 0.0,
            "status": status,
            "reason": reason,
        }
    return summaries
