"""Build an independent exploratory deep-learning dataset from strict supervised artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from funding_arb.config.models import ExploratoryDLDatasetSettings
from funding_arb.utils.paths import ensure_directory, repo_path


@dataclass(frozen=True)
class ExploratoryDLDatasetArtifacts:
    """Files produced by the exploratory dataset builder."""

    output_dir: str
    dataset_path: str
    dataset_csv_path: str | None
    manifest_path: str


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
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _load_optional_json(path_text: str | Path | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _output_dir(settings: ExploratoryDLDatasetSettings) -> Path:
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


def describe_exploratory_dataset_job(
    config: ExploratoryDLDatasetSettings | dict[str, Any]
) -> str:
    """Return a human-readable summary of the exploratory dataset job."""
    settings = (
        config
        if isinstance(config, ExploratoryDLDatasetSettings)
        else ExploratoryDLDatasetSettings.model_validate(config)
    )
    return (
        f"Exploratory DL dataset ready for {settings.input.symbol} on {settings.input.provider} at "
        f"{settings.input.frequency}, reading {settings.input.source_dataset_path} and writing "
        f"independent showcase targets under "
        f"{settings.output.output_dir}/{settings.input.provider}/{settings.input.symbol.lower()}/"
        f"{settings.input.frequency}."
    )


def _derive_targets(
    frame: pd.DataFrame, settings: ExploratoryDLDatasetSettings
) -> pd.DataFrame:
    target = settings.target
    if target.gross_return_column not in frame.columns:
        raise ValueError(
            "Exploratory dataset requires a source gross-return column "
            f"'{target.gross_return_column}'."
        )
    derived = frame.copy()
    gross = pd.to_numeric(derived[target.gross_return_column], errors="coerce")
    short_is_best = gross >= 0.0
    direction_flag = pd.Series(pd.NA, index=derived.index, dtype="Int64")
    direction_flag.loc[gross.notna()] = short_is_best.loc[gross.notna()].astype(int)
    direction_label = pd.Series(pd.NA, index=derived.index, dtype="object")
    direction_label.loc[gross.notna()] = np.where(
        short_is_best.loc[gross.notna()],
        target.short_direction_label,
        target.long_direction_label,
    )

    derived[target.short_direction_gross_column] = gross
    # Under the prototype's equal-notional, zero-borrow gross definition, the
    # opposite direction is the signed mirror image of the default direction.
    derived[target.long_direction_gross_column] = -gross
    derived[target.signed_opportunity_column] = gross
    derived[target.absolute_opportunity_column] = gross.abs()
    derived[target.direction_classification_column] = direction_flag
    derived[target.direction_label_column] = direction_label

    base_ready = (
        pd.Series(derived[target.ready_column], index=derived.index, dtype="boolean")
        if target.ready_column in derived.columns
        else pd.Series(True, index=derived.index, dtype="boolean")
    )
    derived["exploratory_ready"] = (
        base_ready.fillna(False).astype(bool) & gross.notna()
    )
    return derived


def _split_target_summary(
    frame: pd.DataFrame, settings: ExploratoryDLDatasetSettings
) -> dict[str, Any]:
    target = settings.target
    split_column = target.split_column
    if split_column not in frame.columns:
        return {}

    summaries: dict[str, Any] = {}
    for split_name, split_frame in frame.groupby(split_column, dropna=False):
        signed = pd.to_numeric(
            split_frame[target.signed_opportunity_column], errors="coerce"
        ).dropna()
        absolute = pd.to_numeric(
            split_frame[target.absolute_opportunity_column], errors="coerce"
        ).dropna()
        short_best = pd.to_numeric(
            split_frame[target.direction_classification_column], errors="coerce"
        ).dropna()
        summaries[str(split_name)] = {
            "row_count": int(len(split_frame)),
            "ready_count": int(split_frame["exploratory_ready"].fillna(False).astype(bool).sum()),
            "signed_mean_bps": _json_ready(signed.mean() if not signed.empty else None),
            "signed_std_bps": _json_ready(signed.std(ddof=0) if not signed.empty else None),
            "signed_min_bps": _json_ready(signed.min() if not signed.empty else None),
            "signed_max_bps": _json_ready(signed.max() if not signed.empty else None),
            "absolute_mean_bps": _json_ready(absolute.mean() if not absolute.empty else None),
            "absolute_90pct_bps": _json_ready(absolute.quantile(0.9) if not absolute.empty else None),
            "short_direction_rate": _json_ready(short_best.mean() if not short_best.empty else None),
            "long_direction_rate": _json_ready(1.0 - short_best.mean() if not short_best.empty else None),
        }
    return summaries


def run_exploratory_dataset_pipeline(
    settings: ExploratoryDLDatasetSettings,
) -> ExploratoryDLDatasetArtifacts:
    """Create a standalone exploratory dataset without modifying strict labels."""
    dataset = _load_table(settings.input.source_dataset_path)
    source_manifest = _load_optional_json(settings.input.source_manifest_path)
    derived = _derive_targets(dataset, settings)

    output_dir = _output_dir(settings)
    dataset_path = output_dir / settings.output.artifact_name
    dataset_primary_path = _write_frame(derived, dataset_path)
    dataset_csv_path: str | None = None
    if settings.output.write_csv and dataset_path.suffix.lower() != ".csv":
        dataset_csv_path = _write_frame(derived, dataset_path.with_suffix(".csv"))

    manifest = {
        "input": settings.input.model_dump(),
        "target": settings.target.model_dump(),
        "output": settings.output.model_dump(),
        "source_manifest": source_manifest,
        "summary": {
            "row_count": int(len(derived)),
            "column_count": int(len(derived.columns)),
            "split_counts": (
                derived[settings.target.split_column].astype(str).value_counts().to_dict()
                if settings.target.split_column in derived.columns
                else {}
            ),
            "ready_count": int(derived["exploratory_ready"].fillna(False).astype(bool).sum()),
            "target_summary_by_split": _split_target_summary(derived, settings),
        },
        "definitions": {
            "gross_opportunity_target": settings.target.short_direction_gross_column,
            "signed_opportunity_target": settings.target.signed_opportunity_column,
            "direction_classification_target": settings.target.direction_classification_column,
            "direction_label_column": settings.target.direction_label_column,
        },
        "disclaimer": (
            "Exploratory DL targets are supplementary showcase targets derived from the strict supervised dataset. "
            "They are meant to highlight gross-opportunity ranking and direction structure, not to replace the strict "
            "post-cost primary target."
        ),
        "notes": settings.notes,
        "dataset_path": dataset_primary_path,
        "dataset_csv_path": dataset_csv_path,
    }
    manifest_path = output_dir / settings.output.manifest_name
    manifest_path.write_text(
        json.dumps(_json_ready(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ExploratoryDLDatasetArtifacts(
        output_dir=str(output_dir),
        dataset_path=dataset_primary_path,
        dataset_csv_path=dataset_csv_path,
        manifest_path=str(manifest_path),
    )
