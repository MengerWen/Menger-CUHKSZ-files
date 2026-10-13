"""Shared Stage 3 pipeline for characteristic-only benchmark models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.evaluation.metrics import rank_ic_by_month
from src.models.non_graph_benchmarks import CAEStyleBenchmark, IPCAStyleBenchmark, MLPBenchmark, NeuralTrainConfig


@dataclass(frozen=True)
class OOSBlock:
    """One refit block in the common OOS protocol."""

    refit_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_months: int
    validation_months: int
    test_months: int


def load_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML Stage 3 configuration file."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {config_path}")
    return config


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a project-relative path."""

    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_panel_and_metadata(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Load the Stage 2 panel and infer feature columns from metadata."""

    data_config = config.get("data", {})
    panel_path = resolve_project_path(project_root, data_config.get("panel_path", "outputs/panels/main_features500_panel.pkl"))
    metadata_path = resolve_project_path(project_root, data_config.get("metadata_path", "outputs/metadata/main_features500_panel_metadata.json"))
    target_col = data_config.get("target_col", "target_excess_return")

    panel = pd.read_pickle(panel_path)
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_cols = metadata.get("features", {}).get("kept_features")
    if not feature_cols:
        non_features = {
            "date",
            "stock_id",
            "csi500_member_t",
            "target_return",
            "rf_next_month",
            "target_excess_return",
            "mcap_t",
            "blacklisted_t",
            "untradable_t",
        }
        feature_cols = [col for col in panel.select_dtypes(include="number").columns if col not in non_features]

    required = ["date", "stock_id", target_col, *feature_cols]
    missing = [col for col in required if col not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required columns: {missing[:20]}")

    panel = panel[required].copy()
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[target_col, *feature_cols]).sort_values(["date", "stock_id"]).reset_index(drop=True)
    return panel, metadata, list(feature_cols)


def make_oos_blocks(panel: pd.DataFrame, config: dict[str, Any]) -> list[OOSBlock]:
    """Create expanding or rolling OOS blocks shared by all benchmarks."""

    oos_config = config.get("oos", {})
    scheme = oos_config.get("scheme", "expanding")
    if scheme not in {"expanding", "rolling"}:
        raise ValueError("oos.scheme must be either 'expanding' or 'rolling'")

    initial_train_months = int(oos_config.get("initial_train_months", 96))
    validation_months = int(oos_config.get("validation_months", 12))
    refit_frequency_months = int(oos_config.get("refit_frequency_months", 12))
    max_oos_months = oos_config.get("max_oos_months", 24)
    rolling_train_months = int(oos_config.get("rolling_train_months", initial_train_months))

    months = pd.DatetimeIndex(panel["date"].drop_duplicates()).sort_values()
    first_test_idx = initial_train_months + validation_months
    if first_test_idx >= len(months):
        raise ValueError("not enough months for the requested train/validation/OOS split")

    oos_end_idx = len(months)
    if max_oos_months is not None:
        oos_end_idx = min(oos_end_idx, first_test_idx + int(max_oos_months))

    blocks: list[OOSBlock] = []
    refit_id = 0
    for test_start_idx in range(first_test_idx, oos_end_idx, refit_frequency_months):
        test_end_idx = min(test_start_idx + refit_frequency_months, oos_end_idx)
        train_end_idx = test_start_idx - validation_months
        if scheme == "expanding":
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - rolling_train_months)
        validation_start_idx = train_end_idx
        validation_end_idx = test_start_idx
        if train_end_idx <= train_start_idx or validation_end_idx <= validation_start_idx or test_end_idx <= test_start_idx:
            continue

        blocks.append(
            OOSBlock(
                refit_id=refit_id,
                train_start=months[train_start_idx],
                train_end=months[train_end_idx - 1],
                validation_start=months[validation_start_idx],
                validation_end=months[validation_end_idx - 1],
                test_start=months[test_start_idx],
                test_end=months[test_end_idx - 1],
                train_months=train_end_idx - train_start_idx,
                validation_months=validation_end_idx - validation_start_idx,
                test_months=test_end_idx - test_start_idx,
            )
        )
        refit_id += 1
    if not blocks:
        raise ValueError("no OOS blocks were created")
    return blocks


def split_block(panel: pd.DataFrame, block: OOSBlock) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split panel rows into train, validation, and test subsets for a block."""

    train = panel[(panel["date"] >= block.train_start) & (panel["date"] <= block.train_end)].copy()
    validation = panel[(panel["date"] >= block.validation_start) & (panel["date"] <= block.validation_end)].copy()
    test = panel[(panel["date"] >= block.test_start) & (panel["date"] <= block.test_end)].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(f"empty split in refit block {block.refit_id}")
    return train, validation, test


def make_neural_config(config: dict[str, Any], refit_id: int) -> NeuralTrainConfig:
    """Build torch hyperparameters with deterministic per-refit seeds."""

    neural = config.get("neural", {})
    base_seed = int(neural.get("seed", 20260408))
    return NeuralTrainConfig(
        hidden_dim=int(neural.get("hidden_dim", 64)),
        dropout=float(neural.get("dropout", 0.1)),
        learning_rate=float(neural.get("learning_rate", 1.0e-3)),
        weight_decay=float(neural.get("weight_decay", 1.0e-4)),
        batch_size=int(neural.get("batch_size", 8192)),
        max_epochs=int(neural.get("max_epochs", 8)),
        patience=int(neural.get("patience", 2)),
        seed=base_seed + refit_id,
        device=str(neural.get("device", "auto")),
    )


def make_models(config: dict[str, Any], feature_cols: list[str], target_col: str, refit_id: int) -> list[Any]:
    """Instantiate all requested non-graph benchmark models."""

    model_names = config.get("models", {}).get(
        "include",
        ["mlp_predictor", "ipca_style", "conditional_autoencoder_style"],
    )
    n_factors = int(config.get("models", {}).get("latent_dim", 3))
    neural_config = make_neural_config(config, refit_id)
    ipca_config = config.get("ipca", {})

    models: list[Any] = []
    for name in model_names:
        if name == "mlp_predictor":
            models.append(MLPBenchmark(feature_cols, target_col, neural_config))
        elif name == "ipca_style":
            models.append(
                IPCAStyleBenchmark(
                    feature_cols=feature_cols,
                    target_col=target_col,
                    n_factors=n_factors,
                    ridge_alpha=float(ipca_config.get("ridge_alpha", 1.0e-3)),
                    als_iterations=int(ipca_config.get("als_iterations", 2)),
                    seed=neural_config.seed,
                )
            )
        elif name == "conditional_autoencoder_style":
            models.append(CAEStyleBenchmark(feature_cols, target_col, n_factors, neural_config))
        else:
            raise ValueError(f"unknown Stage 3 non-graph model: {name}")
    return models


def add_block_columns(frame: pd.DataFrame, block: OOSBlock, target_col: str) -> pd.DataFrame:
    """Attach refit metadata to a model output frame."""

    out = frame.copy()
    out["refit_id"] = block.refit_id
    out["target_col"] = target_col
    out["train_start"] = block.train_start
    out["train_end"] = block.train_end
    out["validation_start"] = block.validation_start
    out["validation_end"] = block.validation_end
    out["test_start"] = block.test_start
    out["test_end"] = block.test_end
    return out


def prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute prediction and pricing-style diagnostics by model."""

    rows: list[dict[str, Any]] = []
    for model_name, group in predictions.groupby("model", sort=True):
        y_true = group["y_true"].to_numpy(dtype=np.float64)
        y_pred = group["y_pred"].to_numpy(dtype=np.float64)
        residual = y_true - y_pred
        denominator = float(np.sum(y_true**2))
        finance_oos_r2 = np.nan if np.isclose(denominator, 0.0) else 1.0 - float(np.sum(residual**2)) / denominator
        rank_ic = rank_ic_by_month(group, "date", "y_true", "y_pred").dropna()
        monthly_pricing_error = group.assign(residual=residual).groupby("date", sort=True)["residual"].mean()
        rows.append(
            {
                "model": model_name,
                "n_obs": int(group.shape[0]),
                "n_months": int(group["date"].nunique()),
                "finance_oos_r2_zero_benchmark": finance_oos_r2,
                "sklearn_r2_mean_benchmark": float(r2_score(y_true, y_pred)),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                "rank_ic_std": float(rank_ic.std(ddof=1)) if rank_ic.size > 1 else np.nan,
                "rank_ic_tstat": float(rank_ic.mean() / (rank_ic.std(ddof=1) / np.sqrt(rank_ic.size))) if rank_ic.size > 1 and not np.isclose(rank_ic.std(ddof=1), 0.0) else np.nan,
                "pricing_error_monthly_mean": float(monthly_pricing_error.mean()),
                "pricing_error_monthly_rmse": float(np.sqrt(np.mean(monthly_pricing_error.to_numpy(dtype=np.float64) ** 2))),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_stage3_benchmarks(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run the full Stage 3 non-graph benchmark suite."""

    started_at = time.time()
    config = load_config(config_path)
    target_col = config.get("data", {}).get("target_col", "target_excess_return")
    panel, panel_metadata, feature_cols = load_panel_and_metadata(project_root, config)
    blocks = make_oos_blocks(panel, config)

    predictions: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    factors: list[pd.DataFrame] = []

    for block in blocks:
        train, validation, test = split_block(panel, block)
        print(
            f"[refit {block.refit_id}] train {block.train_start.date()}..{block.train_end.date()} "
            f"validation {block.validation_start.date()}..{block.validation_end.date()} "
            f"test {block.test_start.date()}..{block.test_end.date()}"
        )
        for model in make_models(config, feature_cols, target_col, block.refit_id):
            print(f"  fitting {model.model_name}")
            model.fit(train, validation)
            predictions.append(add_block_columns(model.predict(test), block, target_col))
            if hasattr(model, "exposures"):
                exposures.append(add_block_columns(model.exposures(test), block, target_col))
            if hasattr(model, "latent_factors"):
                factors.append(add_block_columns(model.latent_factors(), block, target_col))

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    exposure_frame = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    factor_frame = pd.concat(factors, ignore_index=True) if factors else pd.DataFrame()
    metrics_frame = prediction_metrics(prediction_frame)

    output_config = config.get("outputs", {})
    predictions_path = resolve_project_path(project_root, output_config.get("predictions_path", "outputs/predictions/stage3_non_graph_predictions.pkl"))
    exposures_path = resolve_project_path(project_root, output_config.get("exposures_path", "outputs/latent/stage3_non_graph_exposures.pkl"))
    factors_path = resolve_project_path(project_root, output_config.get("factors_path", "outputs/latent/stage3_non_graph_factors.pkl"))
    metrics_path = resolve_project_path(project_root, output_config.get("metrics_path", "outputs/metrics/stage3_non_graph_metrics.csv"))
    metadata_path = resolve_project_path(project_root, output_config.get("metadata_path", "outputs/metadata/stage3_non_graph_run_metadata.json"))

    for path in [predictions_path, exposures_path, factors_path, metrics_path, metadata_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    prediction_frame.to_pickle(predictions_path)
    exposure_frame.to_pickle(exposures_path)
    factor_frame.to_pickle(factors_path)
    metrics_frame.to_csv(metrics_path, index=False)

    feature_universe = panel_metadata.get("config", {}).get("feature_universe") or panel_metadata.get("feature_universe")
    run_metadata: dict[str, Any] = {
        "stage": "stage3_non_graph_benchmarks",
        "config_path": str(config_path),
        "panel_rows": int(panel.shape[0]),
        "panel_months": int(panel["date"].nunique()),
        "panel_stocks": int(panel["stock_id"].nunique()),
        "panel_start": panel["date"].min(),
        "panel_end": panel["date"].max(),
        "feature_universe": feature_universe,
        "feature_count": len(feature_cols),
        "target_col": target_col,
        "models": config.get("models", {}).get("include"),
        "latent_dim": int(config.get("models", {}).get("latent_dim", 3)),
        "oos_blocks": [asdict(block) for block in blocks],
        "prediction_rows": int(prediction_frame.shape[0]),
        "exposure_rows": int(exposure_frame.shape[0]),
        "factor_rows": int(factor_frame.shape[0]),
        "output_paths": {
            "predictions": str(predictions_path),
            "exposures": str(exposures_path),
            "factors": str(factors_path),
            "metrics": str(metrics_path),
            "metadata": str(metadata_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
        "note": "No graph inputs are used in Stage 3. OOS forecasts use only information from the training and validation window before each test block.",
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, default=_json_default)

    print("Saved Stage 3 outputs:")
    print(f"  predictions: {predictions_path}")
    print(f"  exposures:    {exposures_path}")
    print(f"  factors:      {factors_path}")
    print(f"  metrics:      {metrics_path}")
    print(f"  metadata:     {metadata_path}")
    print(metrics_frame.to_string(index=False))
    return run_metadata

