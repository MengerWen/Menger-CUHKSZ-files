"""Adapters that normalize different strategy/model outputs into a common signal schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from funding_arb.config.models import SignalSettings
from funding_arb.signals.schemas import ensure_signal_schema, validate_prediction_columns
from funding_arb.utils.paths import repo_path



def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)



def _load_prediction_table(path_text: str | Path, source_name: str) -> pd.DataFrame:
    path = _resolve_path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported prediction artifact format: {path.suffix}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    validate_prediction_columns(frame, source_name)
    return frame.sort_values(["timestamp", "model_name"]).reset_index(drop=True)



def _baseline_mode(source_name: str, configured_mode: str) -> str:
    normalized_source = source_name.lower()
    if normalized_source in {"rules", "rule-based", "rule_based"}:
        return "rule_based_only"
    if normalized_source in {"baseline-ml", "baseline_ml", "ml"}:
        return "predictive_only"
    return configured_mode.lower()



def _baseline_subtype(model_family: str) -> str:
    if model_family == "rule_based":
        return "rule_based"
    if model_family == "linear":
        return "baseline_linear"
    if model_family == "tree":
        return "baseline_tree"
    return "baseline_ml"


def _optional_column(frame: pd.DataFrame, column: str, default: Any) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)



def _metadata_json(row: pd.Series, source_name: str, source_path: str) -> str:
    def _safe_metadata_value(value: Any) -> Any:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except TypeError:
            return value
        return value

    metadata = {
        "source_name": source_name,
        "source_path": source_path,
        "candidate_direction": row.get("signal_direction"),
        "raw_decision_score": None if pd.isna(row.get("decision_score")) else float(row.get("decision_score")),
        "signal_threshold": None if pd.isna(row.get("signal_threshold")) else float(row.get("signal_threshold")),
        "signal_strength": None if pd.isna(row.get("signal_strength")) else float(row.get("signal_strength")),
        "predicted_probability": None if pd.isna(row.get("predicted_probability")) else float(row.get("predicted_probability")),
        "predicted_return_bps": None if pd.isna(row.get("predicted_return_bps")) else float(row.get("predicted_return_bps")),
        "predicted_label": None if pd.isna(row.get("predicted_label")) else int(row.get("predicted_label")),
        "selected_threshold_objective": _safe_metadata_value(row.get("selected_threshold_objective")),
        "selected_threshold_objective_value": _safe_metadata_value(row.get("selected_threshold_objective_value")),
        "prediction_mode": _safe_metadata_value(row.get("prediction_mode")),
        "calibration_method": _safe_metadata_value(row.get("calibration_method")),
        "feature_importance_method": _safe_metadata_value(row.get("feature_importance_method")),
        "selected_hyperparameters_json": _safe_metadata_value(row.get("selected_hyperparameters_json")),
        "checkpoint_selection_metric": _safe_metadata_value(row.get("checkpoint_selection_metric")),
        "best_checkpoint_metric_value": _safe_metadata_value(row.get("best_checkpoint_metric_value")),
        "checkpoint_selection_effective_metric": _safe_metadata_value(row.get("checkpoint_selection_effective_metric")),
        "best_checkpoint_effective_metric_value": _safe_metadata_value(row.get("best_checkpoint_effective_metric_value")),
        "checkpoint_selection_fallback_used": _safe_metadata_value(row.get("checkpoint_selection_fallback_used")),
        "selected_loss": _safe_metadata_value(row.get("selected_loss")),
        "regression_loss": _safe_metadata_value(row.get("regression_loss")),
        "use_balanced_classification_loss": _safe_metadata_value(row.get("use_balanced_classification_loss")),
        "preprocessing_scaler": _safe_metadata_value(row.get("preprocessing_scaler")),
        "winsorize_lower_quantile": _safe_metadata_value(row.get("winsorize_lower_quantile")),
        "winsorize_upper_quantile": _safe_metadata_value(row.get("winsorize_upper_quantile")),
        "actual_label": None if pd.isna(row.get("actual_label")) else float(row.get("actual_label")),
        "actual_return_bps": None if pd.isna(row.get("actual_return_bps")) else float(row.get("actual_return_bps")),
    }
    return json.dumps(metadata, ensure_ascii=True, sort_keys=True)



def _normalize_predictions(
    frame: pd.DataFrame,
    settings: SignalSettings,
    source_name: str,
    source_subtype_getter: callable,
    source_path: str,
) -> pd.DataFrame:
    normalized = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
            "asset": settings.input.symbol,
            "venue": settings.input.venue,
            "frequency": settings.input.frequency,
            "source": source_name,
            "source_subtype": frame["model_family"].map(source_subtype_getter),
            "strategy_name": frame["model_name"].astype(str),
            "model_family": frame["model_family"].astype(str),
            "task": frame["task"].astype(str),
            "signal_score": pd.to_numeric(frame["signal_strength"], errors="coerce"),
            "predicted_class": pd.to_numeric(frame["predicted_label"], errors="coerce"),
            "expected_return_bps": pd.to_numeric(frame["predicted_return_bps"], errors="coerce"),
            "signal_threshold": pd.to_numeric(_optional_column(frame, "signal_threshold", np.nan), errors="coerce"),
            "threshold_objective": _optional_column(frame, "selected_threshold_objective", None).astype(object),
            "selected_threshold_objective_value": pd.to_numeric(
                _optional_column(frame, "selected_threshold_objective_value", np.nan), errors="coerce"
            ),
            "prediction_mode": _optional_column(frame, "prediction_mode", None).astype(object),
            "calibration_method": _optional_column(frame, "calibration_method", None).astype(object),
            "feature_importance_method": _optional_column(frame, "feature_importance_method", None).astype(object),
            "selected_hyperparameters_json": _optional_column(frame, "selected_hyperparameters_json", "{}").fillna("{}").astype(str),
            "checkpoint_selection_metric": _optional_column(frame, "checkpoint_selection_metric", None).astype(object),
            "best_checkpoint_metric_value": pd.to_numeric(
                _optional_column(frame, "best_checkpoint_metric_value", np.nan), errors="coerce"
            ),
            "checkpoint_selection_effective_metric": _optional_column(frame, "checkpoint_selection_effective_metric", None).astype(object),
            "best_checkpoint_effective_metric_value": pd.to_numeric(
                _optional_column(frame, "best_checkpoint_effective_metric_value", np.nan), errors="coerce"
            ),
            "checkpoint_selection_fallback_used": pd.Series(
                _optional_column(frame, "checkpoint_selection_fallback_used", False),
                index=frame.index,
                dtype="boolean",
            ).fillna(False).astype(bool),
            "selected_loss": _optional_column(frame, "selected_loss", None).astype(object),
            "regression_loss": _optional_column(frame, "regression_loss", None).astype(object),
            "use_balanced_classification_loss": pd.Series(
                _optional_column(frame, "use_balanced_classification_loss", False),
                index=frame.index,
                dtype="boolean",
            ).fillna(False).astype(bool),
            "preprocessing_scaler": _optional_column(frame, "preprocessing_scaler", None).astype(object),
            "winsorize_lower_quantile": pd.to_numeric(
                _optional_column(frame, "winsorize_lower_quantile", np.nan), errors="coerce"
            ),
            "winsorize_upper_quantile": pd.to_numeric(
                _optional_column(frame, "winsorize_upper_quantile", np.nan), errors="coerce"
            ),
            "suggested_direction": np.where(
                pd.to_numeric(frame["signal"], errors="coerce").fillna(0).astype(int) == 1,
                frame["signal_direction"].fillna("flat").astype(str),
                "flat",
            ),
            "confidence": pd.to_numeric(frame["predicted_probability"], errors="coerce"),
            "should_trade": pd.to_numeric(frame["signal"], errors="coerce").fillna(0).astype(int),
            "split": frame["split"].astype(str),
        }
    )
    normalized["metadata_json"] = frame.apply(_metadata_json, axis=1, source_name=source_name, source_path=source_path)
    return ensure_signal_schema(normalized).sort_values(["timestamp", "strategy_name"]).reset_index(drop=True)



def adapt_baseline_predictions(settings: SignalSettings) -> pd.DataFrame:
    """Normalize baseline prediction artifacts into the unified signal schema."""
    source_name = settings.source.name.lower()
    source_path = str(_resolve_path(settings.input.baseline_predictions_path))
    frame = _load_prediction_table(settings.input.baseline_predictions_path, source_name)
    mode = _baseline_mode(source_name, settings.source.baseline_mode)
    if mode == "rule_based_only":
        frame = frame[frame["model_family"] == "rule_based"].copy()
    elif mode == "predictive_only":
        frame = frame[frame["model_family"] != "rule_based"].copy()
    elif mode != "all":
        raise ValueError(f"Unsupported baseline_mode '{settings.source.baseline_mode}'.")

    if settings.source.model_names:
        allowed = set(settings.source.model_names)
        frame = frame[frame["model_name"].isin(allowed)].copy()

    return _normalize_predictions(
        frame=frame,
        settings=settings,
        source_name=source_name,
        source_subtype_getter=_baseline_subtype,
        source_path=source_path,
    )



def adapt_deep_learning_predictions(settings: SignalSettings) -> pd.DataFrame:
    """Normalize deep-learning prediction artifacts into the unified signal schema."""
    source_name = settings.source.name.lower()
    source_path = str(_resolve_path(settings.input.dl_predictions_path))
    frame = _load_prediction_table(settings.input.dl_predictions_path, source_name)
    if settings.source.model_names:
        allowed = set(settings.source.model_names)
        frame = frame[frame["model_name"].isin(allowed)].copy()
    return _normalize_predictions(
        frame=frame,
        settings=settings,
        source_name=source_name,
        source_subtype_getter=lambda _family: "deep_learning",
        source_path=source_path,
    )
