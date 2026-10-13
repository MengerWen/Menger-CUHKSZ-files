"""Label-generation and supervised-dataset construction pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.config.models import LabelPipelineSettings
from funding_arb.labels.generator import assign_time_series_split, build_label_table
from funding_arb.utils.degeneracy import (
    infer_profitable_column,
    infer_tradeable_column,
    label_split_diagnostics,
)
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class LabelPipelineArtifacts:
    """Paths produced by the label-generation pipeline."""

    supervised_dataset_path: str
    supervised_dataset_csv_path: str | None
    label_table_path: str
    label_table_csv_path: str | None
    split_paths: dict[str, str]
    manifest_path: str


def describe_supervised_dataset_job(config: LabelPipelineSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the label-generation job."""
    settings = config if isinstance(config, LabelPipelineSettings) else LabelPipelineSettings.model_validate(config)
    return (
        f"Label generation ready for {settings.input.symbol} on {settings.input.provider} at {settings.input.frequency}, "
        f"using horizons {settings.target.holding_windows_hours} and writing supervised datasets under "
        f"{settings.output.output_dir}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def _load_table(path: str | Path) -> pd.DataFrame:
    dataset_path = _resolve_path(path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported table format: {dataset_path.suffix}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _output_dir(settings: LabelPipelineSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.output_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def build_supervised_dataset(feature_table: pd.DataFrame, label_table: pd.DataFrame, settings: LabelPipelineSettings) -> pd.DataFrame:
    """Combine features, labels, and time-series split metadata into one modeling table."""
    supervised = feature_table.merge(label_table, on="timestamp", how="left", validate="one_to_one")
    supervised["split"] = assign_time_series_split(supervised["timestamp"], settings.split)
    primary_horizon = settings.target.primary_horizon_hours
    primary_target = f"target_future_net_return_bps_{primary_horizon}h"
    supervised["supervised_ready"] = (
        supervised.get("feature_ready", 0).fillna(0).astype(int).eq(1)
        & supervised[primary_target].notna()
        & supervised["split"].isin(["train", "validation", "test"])
    ).astype(int)
    return supervised


def run_label_pipeline(settings: LabelPipelineSettings) -> LabelPipelineArtifacts:
    """Generate post-cost labels and save supervised datasets ready for modeling."""
    feature_table = _load_table(settings.input.feature_table_path)
    market_table = _load_table(settings.input.market_dataset_path)
    label_table = build_label_table(market_table, settings.target, settings.costs)
    supervised = build_supervised_dataset(feature_table, label_table, settings)

    output_dir = _output_dir(settings)
    supervised_path = output_dir / settings.output.artifact_name
    if supervised_path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("Supervised dataset artifact_name must end with .parquet or .csv")
    label_table_path = output_dir / settings.output.label_table_name
    if label_table_path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("Label table name must end with .parquet or .csv")

    primary_supervised_path = _write_frame(supervised, supervised_path)
    primary_label_path = _write_frame(label_table, label_table_path)

    supervised_csv_path: str | None = None
    label_table_csv_path: str | None = None
    if settings.output.write_csv:
        if supervised_path.suffix.lower() != ".csv":
            supervised_csv_path = _write_frame(supervised, supervised_path.with_suffix(".csv"))
        if label_table_path.suffix.lower() != ".csv":
            label_table_csv_path = _write_frame(label_table, label_table_path.with_suffix(".csv"))

    split_paths: dict[str, str] = {}
    if settings.output.save_split_files:
        split_dir = ensure_directory(output_dir / "splits")
        for split_name in ["train", "validation", "test"]:
            split_frame = supervised[(supervised["split"] == split_name) & (supervised["supervised_ready"] == 1)].copy()
            split_path = split_dir / f"{split_name}.parquet"
            split_paths[split_name] = _write_frame(split_frame, split_path)

    target_columns = [column for column in label_table.columns if column != "timestamp"]
    primary_horizon = settings.target.primary_horizon_hours
    primary_target = f"target_future_net_return_bps_{primary_horizon}h"
    label_diagnostics = label_split_diagnostics(
        supervised,
        split_column="split",
        net_return_column=primary_target,
        tradeable_column=infer_tradeable_column(primary_target),
        profitable_column=infer_profitable_column(primary_target),
        tradeable_threshold_bps=settings.target.min_expected_edge_bps,
        profitable_threshold_bps=settings.target.positive_return_threshold_bps,
    )
    degenerate_summaries = [
        diagnostics
        for split_name, diagnostics in label_diagnostics.items()
        if split_name in {"validation", "test"} and diagnostics.get("status") != "ok"
    ]
    split_counts = {
        split_name: int(((supervised["split"] == split_name) & (supervised["supervised_ready"] == 1)).sum())
        for split_name in ["train", "validation", "test"]
    }
    manifest = {
        "input": settings.input.model_dump(),
        "target": settings.target.model_dump(),
        "costs": settings.costs.model_dump(),
        "split": settings.split.model_dump(),
        "row_count": int(supervised.shape[0]),
        "supervised_ready_rows": int(supervised["supervised_ready"].sum()),
        "primary_target": primary_target,
        "target_columns": target_columns,
        "split_counts": split_counts,
        "label_diagnostics_by_split": label_diagnostics,
        "tradeable_rate_by_split": {
            split_name: diagnostics.get("tradeable_rate")
            for split_name, diagnostics in label_diagnostics.items()
        },
        "profitable_rate_by_split": {
            split_name: diagnostics.get("profitable_rate")
            for split_name, diagnostics in label_diagnostics.items()
        },
        "degenerate_experiment": bool(degenerate_summaries),
        "degenerate_stage": "labels" if degenerate_summaries else None,
        "degenerate_reason": (
            "; ".join(
                f"{split_name}: {diagnostics.get('reason') or diagnostics.get('status')}"
                for split_name, diagnostics in label_diagnostics.items()
                if split_name in {"validation", "test"} and diagnostics.get("status") != "ok"
            )
            if degenerate_summaries
            else None
        ),
        "supervised_dataset_path": primary_supervised_path,
        "supervised_dataset_csv_path": supervised_csv_path,
        "label_table_path": primary_label_path,
        "label_table_csv_path": label_table_csv_path,
        "split_paths": split_paths,
    }
    manifest_path = output_dir / settings.output.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return LabelPipelineArtifacts(
        supervised_dataset_path=primary_supervised_path,
        supervised_dataset_csv_path=supervised_csv_path,
        label_table_path=primary_label_path,
        label_table_csv_path=label_table_csv_path,
        split_paths=split_paths,
        manifest_path=str(manifest_path),
    )
