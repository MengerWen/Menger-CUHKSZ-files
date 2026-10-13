"""Stage 5 training pipeline for graph-enhanced conditional latent factor models."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor, nn
from torch_geometric.data import Data

from src.models.graph_latent_factor import GraphConditionalLatentFactorModel, GraphLatentFactorConfig
from src.training.non_graph_benchmark_pipeline import (
    OOSBlock,
    add_block_columns,
    load_panel_and_metadata,
    make_oos_blocks,
    prediction_metrics,
    resolve_project_path,
    split_block,
)
from src.training.train import get_torch_device, set_global_seed


@dataclass(frozen=True)
class GraphTrainConfig:
    """Training hyperparameters for the Stage 5 graph model."""

    hidden_dim: int = 64
    latent_dim: int = 3
    dropout: float = 0.1
    model_type: str = "gcn"
    gat_heads: int = 2
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    max_epochs: int = 6
    patience: int = 2
    seed: int = 20260408
    device: str = "auto"
    prediction_loss_weight: float = 0.25
    reconstruction_loss_weight: float = 1.0
    pricing_error_weight: float = 0.05
    beta_l2_weight: float = 1.0e-5


@dataclass
class GraphMonthBatch:
    """One monthly full-graph batch."""

    date: pd.Timestamp
    stock_ids: list[str]
    x: Tensor
    edge_index: Tensor
    edge_weight: Tensor | None
    y: Tensor
    mask: Tensor
    date_idx: Tensor | None = None


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config for Stage 5."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {config_path}")
    return config


def project_relative_string(project_root: Path, path: Path | str) -> str:
    """Return project-relative path strings where possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def _device_from_config(device: str) -> torch.device:
    if device == "auto":
        return get_torch_device(prefer_cuda=True)
    return torch.device(device)


def make_train_config(config: dict[str, Any], refit_id: int) -> GraphTrainConfig:
    """Normalize Stage 5 model, optimizer, and loss config."""

    model = config.get("model", {})
    train = config.get("training", {})
    loss = config.get("loss", {})
    base_seed = int(train.get("seed", 20260408))
    return GraphTrainConfig(
        hidden_dim=int(model.get("hidden_dim", 64)),
        latent_dim=int(model.get("latent_dim", 3)),
        dropout=float(model.get("dropout", 0.1)),
        model_type=str(model.get("model_type", "gcn")),
        gat_heads=int(model.get("gat_heads", 2)),
        learning_rate=float(train.get("learning_rate", 1.0e-3)),
        weight_decay=float(train.get("weight_decay", 1.0e-4)),
        max_epochs=int(train.get("max_epochs", 6)),
        patience=int(train.get("patience", 2)),
        seed=base_seed + refit_id,
        device=str(train.get("device", "auto")),
        prediction_loss_weight=float(loss.get("prediction_loss_weight", 0.25)),
        reconstruction_loss_weight=float(loss.get("reconstruction_loss_weight", 1.0)),
        pricing_error_weight=float(loss.get("pricing_error_weight", 0.05)),
        beta_l2_weight=float(loss.get("beta_l2_weight", 1.0e-5)),
    )


def load_graph_manifest(project_root: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Load the Stage 4 graph manifest."""

    manifest_path = resolve_project_path(project_root, config.get("graphs", {}).get("manifest_path", "outputs/graphs/features500_similarity_hybrid_manifest.csv"))
    manifest = pd.read_csv(manifest_path)
    manifest["date"] = pd.to_datetime(manifest["date"])
    required = {"date", "pyg_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"graph manifest missing columns: {sorted(missing)}")
    return manifest.sort_values("date").reset_index(drop=True)


def _load_pyg_graph(project_root: Path, graph_path: str | Path, device: torch.device) -> Data:
    path = resolve_project_path(project_root, graph_path)
    data = torch.load(path, map_location=device, weights_only=False)
    if not hasattr(data, "stock_ids"):
        raise ValueError(f"PyG graph is missing stock_ids metadata: {path}")
    return data.to(device)


def make_graph_month_batch(
    project_root: Path,
    graph_path: str | Path,
    month_frame: pd.DataFrame,
    target_col: str,
    device: torch.device,
    date_idx: int | None = None,
) -> GraphMonthBatch:
    """Join one saved monthly graph with its month-t+1 target vector."""

    data = _load_pyg_graph(project_root, graph_path, device)
    date = pd.Timestamp(month_frame["date"].iloc[0])
    target_by_stock = month_frame.set_index("stock_id")[target_col]
    stock_ids = [str(stock) for stock in data.stock_ids]
    y_np = target_by_stock.reindex(stock_ids).to_numpy(dtype=np.float32)
    mask_np = np.isfinite(y_np)
    y_np = np.nan_to_num(y_np, nan=0.0, posinf=0.0, neginf=0.0)
    if mask_np.sum() == 0:
        raise ValueError(f"no usable targets for graph month {date.date()}")

    edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None
    return GraphMonthBatch(
        date=date,
        stock_ids=stock_ids,
        x=data.x.to(device),
        edge_index=data.edge_index.to(device),
        edge_weight=edge_weight.to(device) if edge_weight is not None else None,
        y=torch.as_tensor(y_np, dtype=torch.float32, device=device),
        mask=torch.as_tensor(mask_np, dtype=torch.bool, device=device),
        date_idx=torch.as_tensor([date_idx], dtype=torch.long, device=device) if date_idx is not None else None,
    )


def load_graph_batches(
    project_root: Path,
    manifest_lookup: dict[pd.Timestamp, str],
    frame: pd.DataFrame,
    target_col: str,
    device: torch.device,
    date_lookup: dict[pd.Timestamp, int] | None = None,
) -> list[GraphMonthBatch]:
    """Load all monthly graph batches for a panel slice."""

    batches: list[GraphMonthBatch] = []
    for date, month_frame in frame.groupby("date", sort=True):
        timestamp = pd.Timestamp(date)
        if timestamp not in manifest_lookup:
            raise KeyError(f"no Stage 4 graph artifact for month {timestamp.date()}")
        date_idx = date_lookup[timestamp] if date_lookup is not None else None
        batches.append(make_graph_month_batch(project_root, manifest_lookup[timestamp], month_frame, target_col, device, date_idx=date_idx))
    return batches


def graph_losses(
    model: GraphConditionalLatentFactorModel,
    batch: GraphMonthBatch,
    config: GraphTrainConfig,
    loss_fn: nn.Module,
) -> tuple[Tensor, dict[str, float]]:
    """Compute configured Stage 5 loss components for one train month."""

    if batch.date_idx is None:
        raise ValueError("train batches require a date_idx for latent factor embeddings")
    reconstruction_pred, beta = model.reconstruct_with_train_factor(batch.x, batch.edge_index, batch.edge_weight, batch.date_idx)
    mean_pred, _beta_mean, _attention = model.predict_with_factor_mean(batch.x, batch.edge_index, batch.edge_weight)
    mask = batch.mask
    reconstruction_loss = loss_fn(reconstruction_pred[mask], batch.y[mask])
    prediction_loss = loss_fn(mean_pred[mask], batch.y[mask])
    residual = reconstruction_pred[mask] - batch.y[mask]
    pricing_error_loss = residual.mean().pow(2)
    beta_l2_loss = beta[mask].pow(2).mean()
    total = (
        config.reconstruction_loss_weight * reconstruction_loss
        + config.prediction_loss_weight * prediction_loss
        + config.pricing_error_weight * pricing_error_loss
        + config.beta_l2_weight * beta_l2_loss
    )
    return total, {
        "total_loss": float(total.detach().cpu()),
        "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
        "prediction_loss": float(prediction_loss.detach().cpu()),
        "pricing_error_loss": float(pricing_error_loss.detach().cpu()),
        "beta_l2_loss": float(beta_l2_loss.detach().cpu()),
    }


def validation_loss(
    model: GraphConditionalLatentFactorModel,
    batches: list[GraphMonthBatch],
    config: GraphTrainConfig,
    loss_fn: nn.Module,
) -> float:
    """Evaluate validation loss using the historical mean latent factor."""

    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            pred, beta, _attention = model.predict_with_factor_mean(batch.x, batch.edge_index, batch.edge_weight)
            mask = batch.mask
            prediction_loss = loss_fn(pred[mask], batch.y[mask])
            pricing_error_loss = (pred[mask] - batch.y[mask]).mean().pow(2)
            beta_l2_loss = beta[mask].pow(2).mean()
            total = prediction_loss + config.pricing_error_weight * pricing_error_loss + config.beta_l2_weight * beta_l2_loss
            losses.append(float(total.detach().cpu()))
    return float(np.mean(losses)) if losses else np.inf


def fit_graph_model(
    project_root: Path,
    manifest_lookup: dict[pd.Timestamp, str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    config: GraphTrainConfig,
) -> tuple[GraphConditionalLatentFactorModel, pd.DatetimeIndex, float]:
    """Fit a graph conditional latent factor model for one OOS refit block."""

    set_global_seed(config.seed)
    device = _device_from_config(config.device)
    train_dates = pd.DatetimeIndex(train["date"].drop_duplicates()).sort_values()
    date_lookup = {date: idx for idx, date in enumerate(train_dates)}
    train_batches = load_graph_batches(project_root, manifest_lookup, train, target_col, device, date_lookup=date_lookup)
    validation_batches = load_graph_batches(project_root, manifest_lookup, validation, target_col, device)

    model_config = GraphLatentFactorConfig(
        input_dim=len(feature_cols),
        hidden_dim=config.hidden_dim,
        n_factors=config.latent_dim,
        dropout=config.dropout,
        model_type=config.model_type,
        gat_heads=config.gat_heads,
    )
    model = GraphConditionalLatentFactorModel(model_config, n_train_dates=len(train_dates)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()

    best_state: dict[str, Tensor] | None = None
    best_loss = np.inf
    bad_epochs = 0
    rng = np.random.default_rng(config.seed)
    for epoch in range(config.max_epochs):
        model.train()
        order = rng.permutation(len(train_batches))
        epoch_losses: list[float] = []
        for pos in order:
            batch = train_batches[int(pos)]
            optimizer.zero_grad(set_to_none=True)
            total_loss, components = graph_losses(model, batch, config, loss_fn)
            total_loss.backward()
            optimizer.step()
            epoch_losses.append(components["total_loss"])

        val_loss = validation_loss(model, validation_batches, config, loss_fn)
        print(f"    epoch {epoch + 1:02d}: train_loss={np.mean(epoch_losses):.6f} validation_loss={val_loss:.6f}")
        if val_loss < best_loss - 1.0e-8:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, train_dates, best_loss


def predict_graph_batches(
    model: GraphConditionalLatentFactorModel,
    batches: list[GraphMonthBatch],
    target_col: str,
    return_attention: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create prediction, exposure, and optional attention frames for graph batches."""

    predictions: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    attentions: list[pd.DataFrame] = []
    model.eval()
    with torch.no_grad():
        for batch in batches:
            pred, beta, attention = model.predict_with_factor_mean(batch.x, batch.edge_index, batch.edge_weight, return_attention=return_attention)
            mask_np = batch.mask.detach().cpu().numpy().astype(bool)
            pred_np = pred.detach().cpu().numpy().astype(np.float64)
            y_np = batch.y.detach().cpu().numpy().astype(np.float64)
            beta_np = beta.detach().cpu().numpy().astype(np.float64)
            stocks = np.asarray(batch.stock_ids, dtype=object)
            date_values = np.repeat(batch.date, mask_np.sum())

            pred_frame = pd.DataFrame(
                {
                    "date": date_values,
                    "stock_id": stocks[mask_np],
                    "model": model.model_name,
                    "y_true": y_np[mask_np],
                    "y_pred": pred_np[mask_np],
                }
            )
            predictions.append(pred_frame)

            exposure_frame = pd.DataFrame({"date": date_values, "stock_id": stocks[mask_np], "model": model.model_name})
            for j in range(beta_np.shape[1]):
                exposure_frame[f"beta_{j + 1}"] = beta_np[mask_np, j]
            exposures.append(exposure_frame)

            if attention is not None:
                edge_index, alpha = attention
                edge_np = edge_index.detach().cpu().numpy()
                alpha_np = alpha.detach().cpu().numpy()
                if alpha_np.ndim == 2:
                    alpha_values = alpha_np.mean(axis=1)
                else:
                    alpha_values = alpha_np
                source_idx = edge_np[0]
                target_idx = edge_np[1]
                attentions.append(
                    pd.DataFrame(
                        {
                            "date": np.repeat(batch.date, len(alpha_values)),
                            "source": stocks[source_idx],
                            "target": stocks[target_idx],
                            "model": model.model_name,
                            "attention_weight": alpha_values.astype(np.float64),
                        }
                    )
                )

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    exposure_frame = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    attention_frame = pd.concat(attentions, ignore_index=True) if attentions else pd.DataFrame(columns=["date", "source", "target", "model", "attention_weight"])
    prediction_frame["target_col"] = target_col
    return prediction_frame, exposure_frame, attention_frame


def latent_factor_frame(model: GraphConditionalLatentFactorModel, train_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Export train-month factor embeddings and forecast mean factor."""

    with torch.no_grad():
        factors_np = model.factor_embeddings.weight.detach().cpu().numpy().astype(np.float64)
        factor_mean = model.factor_mean().detach().cpu().numpy().astype(np.float64)
    frame = pd.DataFrame(factors_np, columns=[f"factor_{j + 1}" for j in range(factors_np.shape[1])])
    frame.insert(0, "date", train_dates)
    frame["model"] = model.model_name
    frame["factor_kind"] = "train_factor"
    mean_row: dict[str, Any] = {"date": pd.NaT, "model": model.model_name, "factor_kind": "forecast_mean"}
    for j, value in enumerate(factor_mean, start=1):
        mean_row[f"factor_{j}"] = value
    return pd.concat([frame, pd.DataFrame([mean_row])], ignore_index=True)


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_stage5_graph_model(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run Stage 5 graph-enhanced conditional latent factor training."""

    started_at = time.time()
    config = load_config(config_path)
    target_col = config.get("data", {}).get("target_col", "target_excess_return")
    panel, panel_metadata, feature_cols = load_panel_and_metadata(project_root, config)
    manifest = load_graph_manifest(project_root, config)
    manifest_lookup = {pd.Timestamp(row.date): row.pyg_path for row in manifest.itertuples(index=False)}
    panel = panel.loc[panel["date"].isin(manifest_lookup)].copy()
    blocks = make_oos_blocks(panel, config)

    all_predictions: list[pd.DataFrame] = []
    all_exposures: list[pd.DataFrame] = []
    all_factors: list[pd.DataFrame] = []
    all_attention: list[pd.DataFrame] = []
    block_summaries: list[dict[str, Any]] = []

    for block in blocks:
        train, validation, test = split_block(panel, block)
        graph_train_config = make_train_config(config, block.refit_id)
        print(
            f"[refit {block.refit_id}] graph model={graph_train_config.model_type} "
            f"train {block.train_start.date()}..{block.train_end.date()} "
            f"validation {block.validation_start.date()}..{block.validation_end.date()} "
            f"test {block.test_start.date()}..{block.test_end.date()}"
        )
        model, train_dates, best_loss = fit_graph_model(project_root, manifest_lookup, train, validation, feature_cols, target_col, graph_train_config)
        device = _device_from_config(graph_train_config.device)
        test_batches = load_graph_batches(project_root, manifest_lookup, test, target_col, device)
        return_attention = graph_train_config.model_type == "gat" and bool(config.get("outputs", {}).get("save_attention", True))
        prediction_frame, exposure_frame, attention_frame = predict_graph_batches(model, test_batches, target_col, return_attention=return_attention)
        factor_frame = latent_factor_frame(model, train_dates)

        all_predictions.append(add_block_columns(prediction_frame, block, target_col))
        all_exposures.append(add_block_columns(exposure_frame, block, target_col))
        all_factors.append(add_block_columns(factor_frame, block, target_col))
        if not attention_frame.empty:
            all_attention.append(add_block_columns(attention_frame, block, target_col))
        block_summaries.append({"refit_id": block.refit_id, "best_validation_loss": best_loss, "train_config": asdict(graph_train_config)})

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    exposures = pd.concat(all_exposures, ignore_index=True) if all_exposures else pd.DataFrame()
    factors = pd.concat(all_factors, ignore_index=True) if all_factors else pd.DataFrame()
    attention = pd.concat(all_attention, ignore_index=True) if all_attention else pd.DataFrame(columns=["date", "source", "target", "model", "attention_weight"])
    metrics = prediction_metrics(predictions)

    outputs = config.get("outputs", {})
    predictions_path = resolve_project_path(project_root, outputs.get("predictions_path", "outputs/predictions/stage5_graph_predictions.pkl"))
    exposures_path = resolve_project_path(project_root, outputs.get("exposures_path", "outputs/latent/stage5_graph_exposures.pkl"))
    factors_path = resolve_project_path(project_root, outputs.get("factors_path", "outputs/latent/stage5_graph_factors.pkl"))
    attention_path = resolve_project_path(project_root, outputs.get("attention_path", "outputs/attention/stage5_graph_attention.pkl"))
    metrics_path = resolve_project_path(project_root, outputs.get("metrics_path", "outputs/metrics/stage5_graph_metrics.csv"))
    metadata_path = resolve_project_path(project_root, outputs.get("metadata_path", "outputs/metadata/stage5_graph_model_metadata.json"))
    for path in [predictions_path, exposures_path, factors_path, attention_path, metrics_path, metadata_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_pickle(predictions_path)
    exposures.to_pickle(exposures_path)
    factors.to_pickle(factors_path)
    attention.to_pickle(attention_path)
    metrics.to_csv(metrics_path, index=False)

    feature_universe = panel_metadata.get("config", {}).get("feature_universe") or panel_metadata.get("feature_universe")
    metadata: dict[str, Any] = {
        "stage": "stage5_graph_conditional_latent_factor",
        "config_path": project_relative_string(project_root, config_path),
        "feature_universe": feature_universe,
        "feature_count": len(feature_cols),
        "target_col": target_col,
        "panel_rows": int(panel.shape[0]),
        "panel_months": int(panel["date"].nunique()),
        "graph_manifest": project_relative_string(project_root, config.get("graphs", {}).get("manifest_path", "outputs/graphs/features500_similarity_hybrid_manifest.csv")),
        "model_config": config.get("model", {}),
        "loss_config": config.get("loss", {}),
        "oos_blocks": [asdict(block) for block in blocks],
        "block_summaries": block_summaries,
        "prediction_rows": int(predictions.shape[0]),
        "exposure_rows": int(exposures.shape[0]),
        "factor_rows": int(factors.shape[0]),
        "attention_rows": int(attention.shape[0]),
        "outputs": {
            "predictions": project_relative_string(project_root, predictions_path),
            "exposures": project_relative_string(project_root, exposures_path),
            "factors": project_relative_string(project_root, factors_path),
            "attention": project_relative_string(project_root, attention_path),
            "metrics": project_relative_string(project_root, metrics_path),
            "metadata": project_relative_string(project_root, metadata_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
        "interpretation": "Graph convolution/attention maps month-t node features and month-t graph context into conditional exposures beta_i,t. Predictions use beta_i,t times the historical mean latent factor premium.",
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=_json_default)

    print("Saved Stage 5 graph model outputs:")
    print(f"  predictions: {predictions_path}")
    print(f"  exposures:   {exposures_path}")
    print(f"  factors:     {factors_path}")
    print(f"  attention:   {attention_path}")
    print(f"  metrics:     {metrics_path}")
    print(f"  metadata:    {metadata_path}")
    print(metrics.to_string(index=False))
    return metadata

