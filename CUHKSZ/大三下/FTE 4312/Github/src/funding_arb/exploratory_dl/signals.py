"""Showcase-oriented signal generation for exploratory deep-learning results."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from funding_arb.config.models import ExploratoryDLSignalRunSettings, ExploratoryDLSignalSettings
from funding_arb.signals.schemas import ensure_signal_schema, validate_prediction_columns
from funding_arb.utils.paths import ensure_directory, repo_path

SHORT_DIRECTION = "short_perp_long_spot"
LONG_DIRECTION = "long_perp_short_spot"


@dataclass(frozen=True)
class ExploratoryDLSignalArtifacts:
    """Files produced by exploratory signal generation."""

    output_dir: str
    signals_path: str
    signals_csv_path: str | None
    manifest_path: str
    strategy_catalog_path: str
    diagnostic_paths: dict[str, str]


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else repo_path(*path.parts)


def _load_table(path_text: str | Path) -> pd.DataFrame:
    path = _resolve_path(path_text)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def _load_optional_json(path_text: str | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def _output_dir(settings: ExploratoryDLSignalSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.output_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / "exploratory"
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if pd.isna(value):
        return None
    return value


def describe_exploratory_signal_job(
    config: ExploratoryDLSignalSettings | dict[str, Any]
) -> str:
    """Return a human-readable summary of the exploratory signal job."""
    settings = (
        config
        if isinstance(config, ExploratoryDLSignalSettings)
        else ExploratoryDLSignalSettings.model_validate(config)
    )
    enabled_runs = [run for run in settings.input.runs if run.enabled]
    return (
        f"Exploratory DL signal generation ready for {len(enabled_runs)} run(s) on "
        f"{settings.input.symbol} ({settings.input.provider}, {settings.input.frequency}), writing "
        f"showcase signals under {settings.output.output_dir}/{settings.input.provider}/"
        f"{settings.input.symbol.lower()}/{settings.input.frequency}/exploratory."
    )


def _run_frame(run: ExploratoryDLSignalRunSettings) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    frame = _load_table(run.prediction_path)
    validate_prediction_columns(frame, f"exploratory_dl:{run.name}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values(["timestamp"]).reset_index(drop=True)
    task = getattr(run, "task", None)
    if task is not None:
        frame["task"] = task
    manifest = _load_optional_json(getattr(run, "manifest_path", None))
    return frame, manifest


def _core_prediction_frame(
    frame: pd.DataFrame,
    run: ExploratoryDLSignalRunSettings,
) -> pd.DataFrame:
    task = str(frame["task"].iloc[0]) if "task" in frame.columns and not frame.empty else (getattr(run, "task", None) or "regression")
    base_model_name = str(frame["model_name"].iloc[0]) if not frame.empty else run.name
    core = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
            "split": frame["split"].astype(str),
            "base_model_name": base_model_name,
            "run_name": run.name,
            "target_type": getattr(run, "target_type", "exploratory_target"),
            "task": task,
            "selected_hyperparameters_json": frame.get("selected_hyperparameters_json", "{}"),
            "prediction_mode": frame.get("prediction_mode"),
            "calibration_method": frame.get("calibration_method"),
            "feature_importance_method": frame.get("feature_importance_method"),
            "checkpoint_selection_metric": frame.get("checkpoint_selection_metric"),
            "best_checkpoint_metric_value": frame.get("best_checkpoint_metric_value"),
            "checkpoint_selection_effective_metric": frame.get("checkpoint_selection_effective_metric"),
            "best_checkpoint_effective_metric_value": frame.get("best_checkpoint_effective_metric_value"),
            "checkpoint_selection_fallback_used": frame.get("checkpoint_selection_fallback_used"),
            "selected_loss": frame.get("selected_loss"),
            "regression_loss": frame.get("regression_loss"),
            "use_balanced_classification_loss": frame.get("use_balanced_classification_loss"),
            "preprocessing_scaler": frame.get("preprocessing_scaler"),
            "winsorize_lower_quantile": frame.get("winsorize_lower_quantile"),
            "winsorize_upper_quantile": frame.get("winsorize_upper_quantile"),
            "actual_return_bps": pd.to_numeric(frame.get("actual_return_bps"), errors="coerce"),
            "actual_label": pd.to_numeric(frame.get("actual_label"), errors="coerce"),
            "predicted_return_bps": pd.to_numeric(frame.get("predicted_return_bps"), errors="coerce"),
            "predicted_probability": pd.to_numeric(frame.get("predicted_probability"), errors="coerce"),
        }
    )
    if task == "classification":
        signed_score = core["predicted_probability"] - 0.5
        magnitude_score = signed_score.abs() * 2.0
        predicted_short = core["predicted_probability"] >= 0.5
        predicted_class = predicted_short.astype(int)
        expected_return_bps = pd.Series(np.nan, index=core.index, dtype=float)
    else:
        signed_score = core["predicted_return_bps"]
        magnitude_score = signed_score.abs()
        predicted_short = signed_score >= 0.0
        predicted_class = predicted_short.astype(int)
        expected_return_bps = magnitude_score.astype(float)
    direction = np.where(predicted_short.fillna(False), SHORT_DIRECTION, LONG_DIRECTION)
    actual_directional_return = np.where(
        direction == SHORT_DIRECTION,
        core["actual_return_bps"],
        -core["actual_return_bps"],
    )

    core["signed_score"] = pd.to_numeric(signed_score, errors="coerce")
    core["absolute_score"] = pd.to_numeric(magnitude_score, errors="coerce")
    core["predicted_class"] = predicted_class.astype(int)
    core["predicted_direction"] = direction
    core["expected_trade_return_bps"] = pd.to_numeric(expected_return_bps, errors="coerce")
    core["actual_directional_return_bps"] = pd.to_numeric(actual_directional_return, errors="coerce")
    return core


def _rolling_threshold(
    absolute_score: pd.Series,
    *,
    percentile_threshold: float,
    window_size: int,
    min_history: int,
) -> pd.Series:
    shifted = absolute_score.shift(1)
    return shifted.rolling(window=window_size, min_periods=min_history).quantile(percentile_threshold)


def _threshold_search_table(
    core: pd.DataFrame,
    *,
    candidate_quantiles: list[float],
    min_signal_count: int,
    min_signal_rate: float,
    support_weight: float,
) -> pd.DataFrame:
    validation = core.loc[core["split"] == "validation"].copy()
    validation = validation.dropna(subset=["absolute_score", "actual_directional_return_bps"])
    if validation.empty:
        return pd.DataFrame(
            [
                {
                    "candidate_quantile": None,
                    "threshold_value": None,
                    "signal_count": 0,
                    "signal_rate": 0.0,
                    "avg_signal_return_bps": None,
                    "cumulative_signal_return_bps": None,
                    "signal_hit_rate": None,
                    "objective_value": None,
                    "support_ok": False,
                    "reason": "validation split has no usable score/return rows",
                }
            ]
        )

    rows: list[dict[str, Any]] = []
    score_series = pd.to_numeric(validation["absolute_score"], errors="coerce").dropna()
    total = len(validation)
    for quantile in candidate_quantiles:
        threshold_value = float(score_series.quantile(float(quantile)))
        signaled = validation.loc[validation["absolute_score"] >= threshold_value].copy()
        returns = pd.to_numeric(signaled["actual_directional_return_bps"], errors="coerce").dropna()
        signal_count = int(len(signaled))
        signal_rate = float(signal_count / total) if total > 0 else 0.0
        avg_return = float(returns.mean()) if not returns.empty else np.nan
        cumulative_return = float(returns.sum()) if not returns.empty else np.nan
        hit_rate = float((returns > 0.0).mean()) if not returns.empty else np.nan
        support_ok = bool(
            signal_count >= min_signal_count and signal_rate >= min_signal_rate and not returns.empty
        )
        objective_value = (
            float(avg_return * math.pow(signal_count, support_weight))
            if support_ok and np.isfinite(avg_return)
            else np.nan
        )
        reason = None
        if signal_count == 0:
            reason = "candidate produced zero validation signals"
        elif signal_count < min_signal_count:
            reason = f"signal_count {signal_count} < min_signal_count {min_signal_count}"
        elif signal_rate < min_signal_rate:
            reason = f"signal_rate {signal_rate:.4f} < min_signal_rate {min_signal_rate:.4f}"
        elif returns.empty:
            reason = "candidate produced no usable validation returns"
        rows.append(
            {
                "candidate_quantile": float(quantile),
                "threshold_value": float(threshold_value),
                "signal_count": signal_count,
                "signal_rate": signal_rate,
                "avg_signal_return_bps": avg_return if np.isfinite(avg_return) else np.nan,
                "cumulative_signal_return_bps": cumulative_return if np.isfinite(cumulative_return) else np.nan,
                "signal_hit_rate": hit_rate if np.isfinite(hit_rate) else np.nan,
                "objective_value": objective_value if np.isfinite(objective_value) else np.nan,
                "support_ok": support_ok,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _summarize_threshold_table(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "status": "no_candidates",
            "reason": "No threshold candidates were generated.",
            "selected_quantile": None,
            "selected_threshold": None,
            "valid_candidate_count": 0,
        }
    valid = frame.loc[frame["support_ok"].fillna(False)].copy()
    if valid.empty:
        return {
            "status": "no_valid_candidates",
            "reason": "All threshold candidates failed the validation support requirements.",
            "selected_quantile": None,
            "selected_threshold": None,
            "valid_candidate_count": 0,
        }
    valid = valid.sort_values(
        ["objective_value", "avg_signal_return_bps", "signal_count"],
        ascending=[False, False, False],
        kind="stable",
        na_position="last",
    )
    best = valid.iloc[0]
    return {
        "status": "ok",
        "reason": None,
        "selected_quantile": _json_ready(best["candidate_quantile"]),
        "selected_threshold": _json_ready(best["threshold_value"]),
        "selected_objective_value": _json_ready(best["objective_value"]),
        "selected_avg_signal_return_bps": _json_ready(best["avg_signal_return_bps"]),
        "selected_signal_count": int(best["signal_count"]),
        "valid_candidate_count": int(len(valid)),
    }


def _strategy_metadata_json(
    row: pd.Series,
    *,
    base_model_name: str,
    run_name: str,
    target_type: str,
    signal_rule: str,
    signal_rule_type: str,
    selection_reason: str | None,
    threshold_summary: dict[str, Any] | None,
) -> str:
    payload = {
        "model_name": base_model_name,
        "run_name": run_name,
        "target_type": target_type,
        "signal_rule": signal_rule,
        "signal_rule_type": signal_rule_type,
        "selection_reason": selection_reason,
        "threshold_search_summary": threshold_summary,
        "raw_signed_score": _json_ready(row.get("signed_score")),
        "absolute_score": _json_ready(row.get("absolute_score")),
        "predicted_direction": row.get("predicted_direction"),
        "actual_directional_return_bps": _json_ready(row.get("actual_directional_return_bps")),
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _build_signal_frame(
    base: pd.DataFrame,
    *,
    settings: ExploratoryDLSignalSettings,
    strategy_name: str,
    signal_rule: str,
    signal_rule_type: str,
    threshold_series: pd.Series,
    should_trade: pd.Series,
    selection_reason: str | None,
    threshold_summary: dict[str, Any] | None,
) -> pd.DataFrame:
    direction = np.where(
        should_trade.fillna(False),
        base["predicted_direction"],
        "flat",
    )
    signal_threshold = pd.to_numeric(threshold_series, errors="coerce")
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(base["timestamp"], utc=True),
            "asset": settings.input.symbol,
            "venue": settings.input.venue,
            "frequency": settings.input.frequency,
            "source": "exploratory_dl",
            "source_subtype": "deep_learning_showcase",
            "strategy_name": strategy_name,
            "model_family": "deep_learning",
            "task": base["task"],
            "signal_score": pd.to_numeric(base["absolute_score"], errors="coerce"),
            "predicted_class": pd.to_numeric(base["predicted_class"], errors="coerce"),
            "expected_return_bps": pd.to_numeric(base["expected_trade_return_bps"], errors="coerce"),
            "signal_threshold": signal_threshold,
            "threshold_objective": selection_reason,
            "selected_threshold_objective_value": (
                threshold_summary.get("selected_objective_value")
                if threshold_summary is not None
                else np.nan
            ),
            "prediction_mode": base["prediction_mode"],
            "calibration_method": base["calibration_method"],
            "feature_importance_method": base["feature_importance_method"],
            "selected_hyperparameters_json": base["selected_hyperparameters_json"].fillna("{}").astype(str),
            "checkpoint_selection_metric": base["checkpoint_selection_metric"],
            "best_checkpoint_metric_value": pd.to_numeric(base["best_checkpoint_metric_value"], errors="coerce"),
            "checkpoint_selection_effective_metric": base["checkpoint_selection_effective_metric"],
            "best_checkpoint_effective_metric_value": pd.to_numeric(
                base["best_checkpoint_effective_metric_value"], errors="coerce"
            ),
            "checkpoint_selection_fallback_used": pd.Series(
                base["checkpoint_selection_fallback_used"], dtype="boolean"
            ).fillna(False).astype(bool),
            "selected_loss": base["selected_loss"],
            "regression_loss": base["regression_loss"],
            "use_balanced_classification_loss": pd.Series(
                base["use_balanced_classification_loss"], dtype="boolean"
            ).fillna(False).astype(bool),
            "preprocessing_scaler": base["preprocessing_scaler"],
            "winsorize_lower_quantile": pd.to_numeric(base["winsorize_lower_quantile"], errors="coerce"),
            "winsorize_upper_quantile": pd.to_numeric(base["winsorize_upper_quantile"], errors="coerce"),
            "suggested_direction": direction,
            "confidence": pd.to_numeric(base["absolute_score"], errors="coerce"),
            "should_trade": should_trade.fillna(False).astype(int),
            "split": base["split"].astype(str),
        }
    )
    frame["metadata_json"] = base.apply(
        _strategy_metadata_json,
        axis=1,
        base_model_name=str(base["base_model_name"].iloc[0]),
        run_name=str(base["run_name"].iloc[0]),
        target_type=str(base["target_type"].iloc[0]),
        signal_rule=signal_rule,
        signal_rule_type=signal_rule_type,
        selection_reason=selection_reason,
        threshold_summary=threshold_summary,
    )
    return ensure_signal_schema(frame)


def _strategy_summary(
    signal_frame: pd.DataFrame,
    *,
    base: pd.DataFrame,
    target_type: str,
    signal_rule: str,
    signal_rule_type: str,
    selection_reason: str | None,
    threshold_summary: dict[str, Any] | None,
    status: str,
    reason: str | None,
) -> dict[str, Any]:
    signal_count_by_split = (
        signal_frame.groupby("split")["should_trade"].sum().astype(int).to_dict()
        if not signal_frame.empty
        else {}
    )
    direction_count = (
        signal_frame.loc[signal_frame["should_trade"] == 1, "suggested_direction"]
        .value_counts()
        .to_dict()
        if not signal_frame.empty
        else {}
    )
    return {
        "strategy_name": str(signal_frame["strategy_name"].iloc[0]),
        "model_name": str(base["base_model_name"].iloc[0]),
        "run_name": str(base["run_name"].iloc[0]),
        "target_type": target_type,
        "task": str(base["task"].iloc[0]),
        "signal_rule": signal_rule,
        "signal_rule_type": signal_rule_type,
        "selection_reason": selection_reason,
        "status": status,
        "reason": reason,
        "signal_count_by_split": signal_count_by_split,
        "direction_count": direction_count,
        "selected_threshold": (
            threshold_summary.get("selected_threshold")
            if threshold_summary is not None
            else None
        ),
        "threshold_search_summary": threshold_summary,
    }


def _build_strategy_frames(
    core: pd.DataFrame,
    run: ExploratoryDLSignalRunSettings,
    settings: ExploratoryDLSignalSettings,
    diagnostics_dir: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, Any]], dict[str, str]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    diagnostic_paths: dict[str, str] = {}

    if settings.ranking_rule.enabled:
        thresholds = _rolling_threshold(
            core["absolute_score"],
            percentile_threshold=settings.ranking_rule.percentile_threshold,
            window_size=settings.ranking_rule.window_size,
            min_history=settings.ranking_rule.min_history,
        )
        should_trade = (
            core["absolute_score"].notna()
            & thresholds.notna()
            & (core["absolute_score"] >= thresholds)
        )
        strategy_name = f"{run.name}__{settings.ranking_rule.name}"
        selection_reason = (
            f"rolling_abs_percentile >= {settings.ranking_rule.percentile_threshold:.2f} "
            f"with window {settings.ranking_rule.window_size} and min_history {settings.ranking_rule.min_history}"
        )
        frame = _build_signal_frame(
            core,
            settings=settings,
            strategy_name=strategy_name,
            signal_rule=settings.ranking_rule.name,
            signal_rule_type="ranking_based",
            threshold_series=thresholds,
            should_trade=should_trade,
            selection_reason=selection_reason,
            threshold_summary=None,
        )
        status = "ok" if int(frame["should_trade"].sum()) > 0 else "no_tradable_signals"
        reason = None if status == "ok" else "rolling ranking rule produced zero active signals"
        frames.append(frame)
        summaries.append(
            _strategy_summary(
                frame,
                base=core,
                target_type=run.target_type,
                signal_rule=settings.ranking_rule.name,
                signal_rule_type="ranking_based",
                selection_reason=selection_reason,
                threshold_summary=None,
                status=status,
                reason=reason,
            )
        )

    if settings.threshold_rule.enabled:
        search_table = _threshold_search_table(
            core,
            candidate_quantiles=settings.threshold_rule.candidate_quantiles,
            min_signal_count=settings.threshold_rule.min_signal_count,
            min_signal_rate=settings.threshold_rule.min_signal_rate,
            support_weight=settings.threshold_rule.support_weight,
        )
        strategy_key = f"{run.name}__{settings.threshold_rule.name}"
        search_path = diagnostics_dir / f"{strategy_key}_threshold_search.csv"
        search_table.to_csv(search_path, index=False)
        diagnostic_paths[f"{strategy_key}_threshold_search"] = str(search_path)
        summary = _summarize_threshold_table(search_table)
        if summary["status"] == "ok":
            threshold_value = float(summary["selected_threshold"])
            thresholds = pd.Series(threshold_value, index=core.index, dtype=float)
            should_trade = (
                core["absolute_score"].notna()
                & (core["absolute_score"] >= threshold_value)
            )
            status = "ok" if int(should_trade.sum()) > 0 else "no_tradable_signals"
            reason = None if status == "ok" else "selected threshold still produced zero active signals"
            selection_reason = (
                f"selected validation absolute-score threshold at quantile {summary['selected_quantile']} "
                "using avg_signal_return_bps weighted by signal support"
            )
        else:
            thresholds = pd.Series(np.nan, index=core.index, dtype=float)
            should_trade = pd.Series(False, index=core.index, dtype=bool)
            status = "no_valid_threshold_candidates"
            reason = summary["reason"]
            selection_reason = summary["reason"]
        strategy_name = f"{run.name}__{settings.threshold_rule.name}"
        frame = _build_signal_frame(
            core,
            settings=settings,
            strategy_name=strategy_name,
            signal_rule=settings.threshold_rule.name,
            signal_rule_type="threshold_based",
            threshold_series=thresholds,
            should_trade=should_trade,
            selection_reason=selection_reason,
            threshold_summary=summary,
        )
        frames.append(frame)
        summaries.append(
            _strategy_summary(
                frame,
                base=core,
                target_type=run.target_type,
                signal_rule=settings.threshold_rule.name,
                signal_rule_type="threshold_based",
                selection_reason=selection_reason,
                threshold_summary=summary,
                status=status,
                reason=reason,
            )
        )

    return frames, summaries, diagnostic_paths


def run_exploratory_signal_generation(
    settings: ExploratoryDLSignalSettings,
) -> ExploratoryDLSignalArtifacts:
    """Generate independent showcase-oriented DL signals."""
    output_dir = _output_dir(settings)
    diagnostics_dir = ensure_directory(output_dir / settings.output.diagnostics_dir_name)

    signal_frames: list[pd.DataFrame] = []
    strategy_summaries: list[dict[str, Any]] = []
    diagnostic_paths: dict[str, str] = {}

    for run in settings.input.runs:
        if not run.enabled:
            continue
        prediction_frame, _manifest = _run_frame(run)
        core = _core_prediction_frame(prediction_frame, run)
        frames, summaries, run_diagnostics = _build_strategy_frames(
            core,
            run,
            settings,
            diagnostics_dir,
        )
        signal_frames.extend(frames)
        strategy_summaries.extend(summaries)
        diagnostic_paths.update(run_diagnostics)

    if not signal_frames:
        raise ValueError("Exploratory signal generation produced no strategy frames.")

    signals = (
        pd.concat(signal_frames, ignore_index=True)
        .sort_values(["strategy_name", "timestamp"])
        .reset_index(drop=True)
    )
    signals_path = output_dir / settings.output.artifact_name
    signals_primary_path = _write_frame(signals, signals_path)
    signals_csv_path: str | None = None
    if settings.output.write_csv and signals_path.suffix.lower() != ".csv":
        signals_csv_path = _write_frame(signals, signals_path.with_suffix(".csv"))

    strategy_catalog = pd.DataFrame(strategy_summaries).sort_values(
        ["model_name", "target_type", "signal_rule"], kind="stable"
    )
    strategy_catalog_path = output_dir / settings.output.strategy_catalog_name
    strategy_catalog_primary_path = _write_frame(strategy_catalog, strategy_catalog_path)

    degenerate = strategy_catalog.loc[
        strategy_catalog["status"].isin(["no_tradable_signals", "no_valid_threshold_candidates"])
    ].copy()
    manifest = {
        "input": settings.input.model_dump(),
        "ranking_rule": settings.ranking_rule.model_dump(),
        "threshold_rule": settings.threshold_rule.model_dump(),
        "output": settings.output.model_dump(),
        "summary": {
            "row_count": int(len(signals)),
            "strategy_count": int(strategy_catalog["strategy_name"].nunique()),
            "active_signal_count": int(signals["should_trade"].sum()),
            "signal_count_by_split": signals.groupby("split")["should_trade"].sum().astype(int).to_dict(),
            "direction_count": (
                signals.loc[signals["should_trade"] == 1, "suggested_direction"]
                .value_counts()
                .to_dict()
            ),
            "nonzero_strategy_count": int(
                sum(
                    any(int(value) > 0 for value in counts.values())
                    for counts in strategy_catalog["signal_count_by_split"]
                )
            ),
            "strategy_summary": strategy_summaries,
        },
        "status": "ok" if degenerate.empty else "warning",
        "reason": (
            None
            if degenerate.empty
            else "; ".join(
                f"{row['strategy_name']}: {row.get('reason') or row['status']}"
                for _, row in degenerate.iterrows()
            )
        ),
        "signals_path": signals_primary_path,
        "signals_csv_path": signals_csv_path,
        "strategy_catalog_path": strategy_catalog_primary_path,
        "diagnostic_paths": diagnostic_paths,
        "disclaimer": (
            "Exploratory DL signals are showcase-oriented translations of independent exploratory model runs. "
            "They use ranking and support-aware threshold rules to demonstrate learned opportunity structure, "
            "and they do not replace the strict post-cost primary signal layer."
        ),
        "notes": settings.notes,
    }
    manifest_path = output_dir / settings.output.manifest_name
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ExploratoryDLSignalArtifacts(
        output_dir=str(output_dir),
        signals_path=signals_primary_path,
        signals_csv_path=signals_csv_path,
        manifest_path=str(manifest_path),
        strategy_catalog_path=strategy_catalog_primary_path,
        diagnostic_paths=diagnostic_paths,
    )
