"""Unified signal-generation pipeline for baseline and deep-learning strategy sources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.config.models import SignalSettings
from funding_arb.signals.adapters import adapt_baseline_predictions, adapt_deep_learning_predictions
from funding_arb.utils.degeneracy import (
    DegenerateExperimentDiagnostics,
    signal_split_diagnostics,
    warn_on_degenerate_experiment,
)
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class SignalArtifacts:
    """Paths produced by the unified signal-generation pipeline."""

    signals_path: str
    signals_csv_path: str | None
    manifest_path: str
    output_dir: str



def describe_signal_job(config: SignalSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the signal-generation job."""
    settings = config if isinstance(config, SignalSettings) else SignalSettings.model_validate(config)
    return (
        f"Signal generation ready for source={settings.source.name} on {settings.input.symbol} "
        f"({settings.input.provider}, {settings.input.frequency}), writing normalized signals under "
        f"{settings.output.output_dir}/{settings.input.provider}/{settings.input.symbol.lower()}/"
        f"{settings.input.frequency}/{settings.source.name.lower()}."
    )



def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)



def _output_dir(settings: SignalSettings) -> Path:
    source_dir = settings.source.name.lower().replace(" ", "_")
    return ensure_directory(
        _resolve_path(settings.output.output_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / source_dir
    )



def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def _load_json(path_text: str | Path | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _source_manifest(settings: SignalSettings, source_name: str) -> dict[str, Any] | None:
    normalized = source_name.lower()
    if normalized == "dl":
        return _load_json(settings.input.dl_manifest_path)
    return _load_json(settings.input.baseline_manifest_path)


def _source_strategy_metadata(
    settings: SignalSettings,
    frame: pd.DataFrame,
    source_name: str,
) -> dict[str, dict[str, Any]]:
    manifest = _source_manifest(settings, source_name)
    if manifest is None:
        return {}
    normalized = source_name.lower()
    if normalized == "dl":
        base_summary = {
            "status": manifest.get("status") or ("ok" if not manifest.get("degenerate_experiment") else "warning"),
            "reason": manifest.get("degenerate_reason"),
            "degenerate_experiment": bool(manifest.get("degenerate_experiment", False)),
            "fallback_used": bool(manifest.get("fallback_used", False)),
            "fallback_reason": manifest.get("fallback_reason"),
            "signal_count_by_split": manifest.get("signal_count_by_split", {}),
            "tradeable_rate_by_split": manifest.get("tradeable_rate_by_split", {}),
            "profitable_rate_by_split": manifest.get("profitable_rate_by_split", {}),
            "selected_threshold": manifest.get("selected_threshold"),
            "threshold_search_summary": manifest.get("threshold_search_summary"),
        }
        return {
            strategy_name: dict(base_summary)
            for strategy_name in frame["strategy_name"].dropna().astype(str).unique().tolist()
        }
    summaries = manifest.get("model_summary", [])
    return {
        str(entry.get("model_name")): {
            "status": entry.get("status", "ok"),
            "reason": entry.get("reason"),
            "degenerate_experiment": bool(entry.get("degenerate_experiment", False)),
            "fallback_used": bool(entry.get("fallback_used", False)),
            "fallback_reason": entry.get("fallback_reason"),
            "signal_count_by_split": entry.get("signal_count_by_split", {}),
            "tradeable_rate_by_split": entry.get("tradeable_rate_by_split", {}),
            "profitable_rate_by_split": entry.get("profitable_rate_by_split", {}),
            "selected_threshold": entry.get("selected_threshold"),
            "threshold_search_summary": entry.get("threshold_search_summary"),
        }
        for entry in summaries
        if entry.get("model_name") is not None
    }


def _strategy_status_summary(
    frame: pd.DataFrame,
    *,
    settings: SignalSettings,
    source_name: str,
) -> list[dict[str, Any]]:
    upstream_metadata = _source_strategy_metadata(settings, frame, source_name)
    summaries: list[dict[str, Any]] = []
    for strategy_name, strategy_frame in frame.groupby("strategy_name", sort=True):
        split_summary = signal_split_diagnostics(
            strategy_frame,
            split_names=strategy_frame["split"].dropna().astype(str).unique().tolist(),
        )
        strategy_signal_count_by_split = {
            split_name: int(diagnostics.get("signal_count", 0))
            for split_name, diagnostics in split_summary.items()
        }
        flagged = [
            (split_name, diagnostics)
            for split_name, diagnostics in split_summary.items()
            if split_name in {"validation", "test"} and diagnostics.get("status") != "ok"
        ]
        status = "ok"
        reason = None
        if flagged:
            status = "no_tradable_signals"
            reason = "; ".join(
                f"{split_name}: {diagnostics.get('reason') or diagnostics.get('status')}"
                for split_name, diagnostics in flagged
            )
            diagnostics = DegenerateExperimentDiagnostics(
                stage="signal_generation",
                reason=reason,
                status=status,
                strategy_name=str(strategy_name),
                source=source_name,
                signal_count_by_split=strategy_signal_count_by_split,
            )
            warn_on_degenerate_experiment(diagnostics, stacklevel=3)
        upstream = upstream_metadata.get(str(strategy_name), {})
        if not flagged and upstream.get("degenerate_experiment"):
            status = str(upstream.get("status") or "degenerate")
            reason = upstream.get("reason")
        threshold_values = pd.to_numeric(strategy_frame.get("signal_threshold"), errors="coerce").dropna()
        selected_threshold = (
            float(threshold_values.iloc[0])
            if not threshold_values.empty and threshold_values.nunique(dropna=True) == 1
            else upstream.get("selected_threshold")
        )
        summaries.append(
            {
                "strategy_name": str(strategy_name),
                "status": status,
                "reason": reason or upstream.get("reason"),
                "degenerate_experiment": bool(flagged) or bool(upstream.get("degenerate_experiment", False)),
                "fallback_used": bool(upstream.get("fallback_used", False)),
                "fallback_reason": upstream.get("fallback_reason"),
                "signal_count_by_split": strategy_signal_count_by_split,
                "tradeable_rate_by_split": upstream.get("tradeable_rate_by_split", {}),
                "profitable_rate_by_split": upstream.get("profitable_rate_by_split", {}),
                "selected_threshold": selected_threshold,
                "threshold_search_summary": upstream.get("threshold_search_summary"),
            }
        )
    return summaries



def _signal_summary(frame: pd.DataFrame, *, settings: SignalSettings) -> dict[str, Any]:
    strategy_columns = [
        "strategy_name",
        "source_subtype",
        "model_family",
        "task",
        "signal_threshold",
        "threshold_objective",
        "selected_threshold_objective_value",
        "prediction_mode",
        "calibration_method",
        "feature_importance_method",
        "checkpoint_selection_metric",
        "checkpoint_selection_effective_metric",
        "checkpoint_selection_fallback_used",
        "selected_loss",
        "regression_loss",
        "preprocessing_scaler",
    ]
    strategy_summary = (
        frame[strategy_columns]
        .drop_duplicates()
        .sort_values(["strategy_name", "source_subtype"])
        .reset_index(drop=True)
    )
    source_name = str(frame["source"].iloc[0]) if not frame.empty else "unknown"
    merged_summary = strategy_summary.to_dict(orient="records")
    status_summary = (
        _strategy_status_summary(frame, settings=settings, source_name=source_name)
        if not frame.empty
        else []
    )
    status_by_strategy = {
        entry["strategy_name"]: entry for entry in status_summary
    }
    for entry in merged_summary:
        status_entry = status_by_strategy.get(str(entry.get("strategy_name")), {})
        entry.update(status_entry)
    overall_signal_counts = signal_split_diagnostics(frame)
    signal_count_by_split = {
        split_name: int(diagnostics.get("signal_count", 0))
        for split_name, diagnostics in overall_signal_counts.items()
    }
    degenerate_entries = [entry for entry in status_summary if entry.get("degenerate_experiment")]
    return {
        "row_count": int(len(frame)),
        "active_signal_count": int(frame["should_trade"].sum()),
        "active_signal_rate": float(frame["should_trade"].mean()) if not frame.empty else 0.0,
        "status": "ok" if not degenerate_entries else "warning",
        "reason": (
            "; ".join(
                f"{entry['strategy_name']}: {entry.get('reason') or entry.get('status')}"
                for entry in degenerate_entries
            )
            if degenerate_entries
            else None
        ),
        "degenerate_experiment": bool(degenerate_entries),
        "signal_count_by_split": signal_count_by_split,
        "selected_threshold": {
            entry["strategy_name"]: entry.get("selected_threshold")
            for entry in status_summary
        },
        "threshold_search_summary": {
            entry["strategy_name"]: entry.get("threshold_search_summary")
            for entry in status_summary
        },
        "strategies": sorted(frame["strategy_name"].dropna().astype(str).unique().tolist()),
        "source_subtypes": sorted(frame["source_subtype"].dropna().astype(str).unique().tolist()),
        "prediction_modes": sorted(frame["prediction_mode"].dropna().astype(str).unique().tolist()),
        "calibration_methods": sorted(frame["calibration_method"].dropna().astype(str).unique().tolist()),
        "feature_importance_methods": sorted(frame["feature_importance_method"].dropna().astype(str).unique().tolist()),
        "threshold_objectives": sorted(frame["threshold_objective"].dropna().astype(str).unique().tolist()),
        "checkpoint_selection_metrics": sorted(frame["checkpoint_selection_metric"].dropna().astype(str).unique().tolist()),
        "checkpoint_selection_effective_metrics": sorted(
            frame["checkpoint_selection_effective_metric"].dropna().astype(str).unique().tolist()
        ),
        "selected_losses": sorted(frame["selected_loss"].dropna().astype(str).unique().tolist()),
        "regression_losses": sorted(frame["regression_loss"].dropna().astype(str).unique().tolist()),
        "preprocessing_scalers": sorted(frame["preprocessing_scaler"].dropna().astype(str).unique().tolist()),
        "checkpoint_selection_fallback_count": int(frame["checkpoint_selection_fallback_used"].fillna(False).astype(bool).sum()),
        "splits": frame["split"].value_counts().to_dict(),
        "directions": frame["suggested_direction"].value_counts().to_dict(),
        "strategy_summary": merged_summary,
    }



def run_signal_generation(settings: SignalSettings) -> SignalArtifacts:
    """Normalize one strategy/model source into the shared signal artifact format."""
    source_name = settings.source.name.lower()
    if source_name in {"baseline", "rules", "rule-based", "rule_based", "baseline-ml", "baseline_ml", "ml"}:
        signals = adapt_baseline_predictions(settings)
    elif source_name == "dl":
        signals = adapt_deep_learning_predictions(settings)
    else:
        raise ValueError(
            f"Unsupported signal source '{settings.source.name}'. Use one of: baseline, rules, baseline-ml, dl."
        )

    output_dir = _output_dir(settings)
    signals_path = output_dir / settings.output.artifact_name
    if signals_path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("Signal artifact_name must end with .parquet or .csv")
    signals_primary_path = _write_frame(signals, signals_path)

    signals_csv_path: str | None = None
    if settings.output.write_csv and signals_path.suffix.lower() != ".csv":
        signals_csv_path = _write_frame(signals, signals_path.with_suffix(".csv"))

    summary = _signal_summary(signals, settings=settings)
    manifest = {
        "input": settings.input.model_dump(),
        "source": settings.source.model_dump(),
        "output": settings.output.model_dump(),
        "summary": summary,
        "status": summary.get("status"),
        "reason": summary.get("reason"),
        "degenerate_experiment": summary.get("degenerate_experiment"),
        "signal_count_by_split": summary.get("signal_count_by_split"),
        "selected_threshold": summary.get("selected_threshold"),
        "threshold_search_summary": summary.get("threshold_search_summary"),
        "signals_path": signals_primary_path,
        "signals_csv_path": signals_csv_path,
    }
    manifest_path = output_dir / settings.output.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SignalArtifacts(
        signals_path=signals_primary_path,
        signals_csv_path=signals_csv_path,
        manifest_path=str(manifest_path),
        output_dir=str(output_dir),
    )
