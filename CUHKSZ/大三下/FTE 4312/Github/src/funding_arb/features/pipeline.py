"""Feature engineering pipeline for perpetual funding-rate arbitrage research."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.config.models import FeatureSettings
from funding_arb.features.builders import (
    build_basis_features,
    build_funding_features,
    build_interaction_state_features,
    build_liquidity_features,
    build_volatility_features,
)
from funding_arb.utils.paths import ensure_directory, repo_path

BASE_COLUMNS = [
    "timestamp",
    "symbol",
    "venue",
    "frequency",
    "perp_close",
    "spot_close",
    "funding_rate",
    "funding_event",
    "perp_volume",
    "spot_volume",
    "open_interest",
    "perp_close_was_missing",
    "spot_close_was_missing",
    "open_interest_was_missing",
]


@dataclass(frozen=True)
class FeaturePipelineArtifacts:
    """Paths produced by a full feature-engineering run."""

    feature_table_path: str
    feature_table_csv_path: str | None
    manifest_path: str


def describe_feature_job(config: FeatureSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the feature job."""
    settings = config if isinstance(config, FeatureSettings) else FeatureSettings.model_validate(config)
    windows = settings.feature_set.rolling_windows
    return (
        f"Feature engineering ready for {settings.input.symbol} at {settings.input.frequency}, "
        f"using rolling windows {windows} from {settings.input.dataset_path} and writing outputs under "
        f"{settings.output.processed_dir}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def load_canonical_dataset(path: str | Path) -> pd.DataFrame:
    """Load the canonical hourly market dataset from parquet or CSV."""
    dataset_path = _resolve_path(path)
    if dataset_path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif dataset_path.suffix.lower() == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _available_base_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in BASE_COLUMNS if column in frame.columns]


def _feature_output_dir(settings: FeatureSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.processed_dir)
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
        raise ValueError(f"Unsupported feature output format: {path.suffix}")
    return str(path)


def build_feature_table(frame: pd.DataFrame, settings: FeatureSettings) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build the final feature table and group-wise feature catalog."""
    feature_settings = settings.feature_set
    funding = build_funding_features(frame, feature_settings)
    basis = build_basis_features(frame, feature_settings)
    volatility = build_volatility_features(frame, feature_settings)
    liquidity = build_liquidity_features(frame, feature_settings)
    interaction = build_interaction_state_features(frame, feature_settings, funding, basis, volatility, liquidity)

    group_columns = {
        "funding": list(funding.columns),
        "basis": list(basis.columns),
        "volatility": list(volatility.columns),
        "liquidity": list(liquidity.columns),
        "interaction_state": list(interaction.columns),
    }

    features = pd.concat([funding, basis, volatility, liquidity, interaction], axis=1)
    base = frame[_available_base_columns(frame)].copy()
    result = pd.concat([base, features], axis=1)

    readiness_columns = [
        f"funding_zscore_{feature_settings.zscore_window}h",
        f"spread_zscore_{feature_settings.zscore_window}h",
        f"perp_realized_vol_{feature_settings.volatility_window}h",
        f"perp_volume_ratio_{feature_settings.liquidity_window}h",
    ]
    available_readiness = [column for column in readiness_columns if column in result.columns]
    result["feature_ready"] = result[available_readiness].notna().all(axis=1).astype(int) if available_readiness else 0
    return result, group_columns


def run_feature_pipeline(settings: FeatureSettings) -> FeaturePipelineArtifacts:
    """Load the canonical dataset, build grouped features, and persist the final table."""
    market_data = load_canonical_dataset(settings.input.dataset_path)
    feature_table, group_columns = build_feature_table(market_data, settings)

    output_dir = _feature_output_dir(settings)
    feature_table_path = output_dir / settings.output.artifact_name
    if feature_table_path.suffix.lower() not in {".parquet", ".csv"}:
        raise ValueError("Feature artifact_name must end with .parquet or .csv")

    primary_path = _write_frame(feature_table, feature_table_path)
    csv_path: str | None = None
    if settings.output.write_csv and feature_table_path.suffix.lower() != ".csv":
        csv_path = _write_frame(feature_table, feature_table_path.with_suffix(".csv"))

    manifest_path = output_dir / settings.output.manifest_name
    manifest = {
        "input": settings.input.model_dump(),
        "feature_set": settings.feature_set.model_dump(),
        "labels": settings.labels.model_dump(),
        "row_count": int(feature_table.shape[0]),
        "feature_ready_rows": int(feature_table["feature_ready"].sum()),
        "base_columns": _available_base_columns(market_data),
        "feature_group_columns": group_columns,
        "feature_count": int(sum(len(columns) for columns in group_columns.values())),
        "feature_table_path": primary_path,
        "feature_table_csv_path": csv_path,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return FeaturePipelineArtifacts(
        feature_table_path=primary_path,
        feature_table_csv_path=csv_path,
        manifest_path=str(manifest_path),
    )