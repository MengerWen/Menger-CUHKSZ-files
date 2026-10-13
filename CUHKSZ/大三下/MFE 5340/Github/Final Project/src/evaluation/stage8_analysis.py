"""Stage 8 interpretability, robustness, and report-ready analysis."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml

from src.evaluation.model_comparison import (
    align_common_prediction_panel,
    monthly_metric_table,
    project_relative_string,
    summary_metric_table,
)
from src.graphs.build_graph import edges_to_pyg_data
from src.graphs.monthly_graphs import load_yaml_config as load_stage4_yaml_config
from src.graphs.monthly_graphs import run_stage4_graph_construction
from src.portfolio.backtest import (
    align_common_signal_panel,
    build_all_weights,
    compute_monthly_portfolio_returns,
    expand_transaction_cost_scenarios,
    load_config as load_stage7_config,
    merge_signal_inputs,
    plot_cumulative_returns,
    summarize_performance,
)
from src.training.graph_model_pipeline import (
    GraphMonthBatch,
    _device_from_config,
    fit_graph_model,
    latent_factor_frame,
    load_config as load_stage5_yaml_config,
    load_graph_batches,
    load_graph_manifest,
    make_train_config,
    predict_graph_batches,
    run_stage5_graph_model,
)
from src.training.non_graph_benchmark_pipeline import (
    add_block_columns,
    load_panel_and_metadata,
    make_oos_blocks,
    resolve_project_path,
    split_block,
)

sns.set_theme(style="whitegrid")


@dataclass
class DiagnosticBlockRun:
    """One fitted Stage 5 refit block kept in memory for diagnostics."""

    refit_id: int
    model: torch.nn.Module
    test_batches: list[GraphMonthBatch]
    block: Any
    model_type: str
    feature_cols: list[str]


def load_yaml(config_path: Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {config_path}")
    return config


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge config dictionaries."""

    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def save_yaml_config(config: dict[str, Any], path: Path) -> None:
    """Persist one generated YAML config."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_stage8_inputs(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Load the core Stage 2, 3, 5, 6, and 7 artifacts used in Stage 8."""

    data_cfg = config.get("data", {})
    panel_path = resolve_project_path(project_root, data_cfg.get("panel_path", "outputs/panels/main_features500_panel.pkl"))
    metadata_path = resolve_project_path(project_root, data_cfg.get("metadata_path", "outputs/metadata/main_features500_panel_metadata.json"))
    main_cfg = config.get("main_results", {})

    panel = pd.read_pickle(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    with metadata_path.open("r", encoding="utf-8") as file:
        panel_metadata = json.load(file)
    feature_cols = panel_metadata.get("features", {}).get("kept_features", [])
    if not feature_cols:
        raise ValueError("Stage 8 could not infer kept Stage 2 features from metadata")

    stage3_pred = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage3_predictions_path"])).copy()
    stage3_exp = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage3_exposures_path"])).copy()
    stage3_fac = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage3_factors_path"])).copy()
    stage5_pred = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage5_predictions_path"])).copy()
    stage5_exp = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage5_exposures_path"])).copy()
    stage5_fac = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage5_factors_path"])).copy()
    stage5_att = pd.read_pickle(resolve_project_path(project_root, main_cfg["stage5_attention_path"])).copy()
    stage6_summary = pd.read_csv(resolve_project_path(project_root, main_cfg["stage6_summary_path"]))
    stage7_perf = pd.read_csv(resolve_project_path(project_root, main_cfg["stage7_performance_path"]))
    graph_manifest = pd.read_csv(resolve_project_path(project_root, main_cfg["graph_manifest_path"]))

    for frame in [stage3_pred, stage3_exp, stage3_fac, stage5_pred, stage5_exp, stage5_fac, stage5_att]:
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"])
    graph_manifest["date"] = pd.to_datetime(graph_manifest["date"])

    main_predictions = pd.concat([stage3_pred, stage5_pred], ignore_index=True)
    main_exposures = pd.concat([stage3_exp, stage5_exp], ignore_index=True)
    main_factors = pd.concat([stage3_fac, stage5_fac], ignore_index=True)
    aligned_predictions, coverage, month_counts = align_common_prediction_panel(main_predictions)
    return {
        "panel": panel,
        "panel_metadata": panel_metadata,
        "feature_cols": feature_cols,
        "stage3_predictions": stage3_pred,
        "stage3_exposures": stage3_exp,
        "stage3_factors": stage3_fac,
        "stage5_predictions": stage5_pred,
        "stage5_exposures": stage5_exp,
        "stage5_factors": stage5_fac,
        "stage5_attention": stage5_att,
        "stage6_summary": stage6_summary,
        "stage7_performance": stage7_perf,
        "graph_manifest": graph_manifest,
        "main_predictions": main_predictions,
        "main_exposures": main_exposures,
        "main_factors": main_factors,
        "aligned_predictions": aligned_predictions,
        "coverage": coverage,
        "month_counts": month_counts,
    }


def feature_exposure_association(
    panel: pd.DataFrame,
    exposures: pd.DataFrame,
    feature_cols: list[str],
    model_names: list[str],
) -> pd.DataFrame:
    """Summarize monthly feature-beta associations for models with saved exposures."""

    beta_cols = [col for col in exposures.columns if col.startswith("beta_")]
    merged = exposures.loc[exposures["model"].isin(model_names), ["date", "stock_id", "model", *beta_cols]].merge(
        panel[["date", "stock_id", *feature_cols]],
        on=["date", "stock_id"],
        how="inner",
        validate="many_to_one",
    )

    rows: list[dict[str, Any]] = []
    for model in model_names:
        model_frame = merged.loc[merged["model"] == model].copy()
        for feature in feature_cols:
            for beta_col in beta_cols:
                monthly = (
                    model_frame.groupby("date", sort=True)
                    .apply(lambda block: block[feature].corr(block[beta_col], method="spearman"), include_groups=False)
                    .dropna()
                )
                if monthly.empty:
                    continue
                rows.append(
                    {
                        "model": model,
                        "feature": feature,
                        "beta": beta_col,
                        "mean_spearman": float(monthly.mean()),
                        "mean_abs_spearman": float(monthly.abs().mean()),
                        "median_abs_spearman": float(monthly.abs().median()),
                        "n_months": int(monthly.size),
                    }
                )
    return pd.DataFrame(rows).sort_values(["model", "mean_abs_spearman"], ascending=[True, False]).reset_index(drop=True)

def summarize_top_feature_links(association: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select report-ready top feature links by model."""

    if association.empty:
        return association
    top = (
        association.groupby(["model", "feature"], as_index=False)
        .agg(
            strongest_beta=("beta", "first"),
            max_abs_spearman=("mean_abs_spearman", "max"),
            avg_abs_spearman=("mean_abs_spearman", "mean"),
        )
        .sort_values(["model", "max_abs_spearman"], ascending=[True, False])
        .groupby("model", group_keys=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return top


def load_edge_payload_lookup(project_root: Path, manifest: pd.DataFrame) -> dict[pd.Timestamp, Path]:
    """Map each graph month to its saved edge payload path."""

    required = {"date", "edge_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"graph manifest missing required columns: {sorted(missing)}")
    return {
        pd.Timestamp(row.date): resolve_project_path(project_root, row.edge_path)
        for row in manifest.itertuples(index=False)
    }


def graph_neighbor_summary(
    project_root: Path,
    manifest: pd.DataFrame,
    graph_predictions: pd.DataFrame,
    top_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize graph structure around top-ranked graph-model names."""

    edge_lookup = load_edge_payload_lookup(project_root, manifest)
    rows: list[dict[str, Any]] = []
    edge_mix_rows: list[pd.DataFrame] = []
    for date, month_pred in graph_predictions.groupby("date", sort=True):
        date = pd.Timestamp(date)
        if date not in edge_lookup:
            continue
        payload = pd.read_pickle(edge_lookup[date])
        typed_edges = payload["typed_edges"].copy()
        combined_edges = payload["combined_edges"].copy()
        if combined_edges.empty:
            continue

        ranked = month_pred.sort_values(["y_pred", "stock_id"], ascending=[False, True]).reset_index(drop=True)
        n_top = max(10, int(np.floor(len(ranked) * float(top_quantile))))
        selected = set(ranked.head(n_top)["stock_id"].astype(str))
        universe = pd.Index(payload["node_features"].index.astype(str))

        degree = pd.concat(
            [
                combined_edges["source"].value_counts(),
                combined_edges["target"].value_counts(),
            ],
            axis=1,
        ).fillna(0.0).sum(axis=1)
        degree = degree.reindex(universe, fill_value=0.0)

        touching_top = combined_edges.loc[
            combined_edges["source"].isin(selected) | combined_edges["target"].isin(selected)
        ].copy()
        inside_top = combined_edges.loc[
            combined_edges["source"].isin(selected) & combined_edges["target"].isin(selected)
        ].copy()

        rows.append(
            {
                "date": date,
                "n_universe": int(len(universe)),
                "n_top": int(len(selected)),
                "mean_degree_universe": float(degree.mean()),
                "mean_degree_top": float(degree.reindex(list(selected), fill_value=0.0).mean()) if selected else np.nan,
                "mean_weight_universe": float(combined_edges["weight"].mean()),
                "mean_weight_touching_top": float(touching_top["weight"].mean()) if not touching_top.empty else np.nan,
                "top_edge_density_share": float(len(inside_top) / len(touching_top)) if len(touching_top) else np.nan,
            }
        )

        if not typed_edges.empty and selected:
            top_typed = typed_edges.loc[
                typed_edges["source"].isin(selected) | typed_edges["target"].isin(selected)
            ].copy()
            if not top_typed.empty:
                edge_mix = top_typed.groupby("edge_type", sort=True)["weight"].agg(["count", "mean"]).reset_index()
                edge_mix["date"] = date
                edge_mix["share"] = edge_mix["count"] / edge_mix["count"].sum()
                edge_mix_rows.append(edge_mix.rename(columns={"count": "edge_count", "mean": "mean_weight"}))

    monthly = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    edge_mix = pd.concat(edge_mix_rows, ignore_index=True) if edge_mix_rows else pd.DataFrame()
    return monthly, edge_mix


def select_permutation_candidates(
    association: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    top_n: int,
) -> list[str]:
    """Choose focused permutation candidates from exposure-feature links."""

    if association.empty:
        return feature_cols[:top_n]
    ranked = (
        association.loc[association["model"] == model_name]
        .groupby("feature", as_index=False)["mean_abs_spearman"]
        .max()
        .sort_values("mean_abs_spearman", ascending=False)
    )
    features = ranked["feature"].head(top_n).tolist()
    if len(features) < top_n:
        extras = [feature for feature in feature_cols if feature not in features]
        features.extend(extras[: top_n - len(features)])
    return features


def summarize_prediction_frame(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned coverage, monthly metrics, and summary metrics."""

    aligned, coverage, _month_counts = align_common_prediction_panel(predictions)
    monthly = monthly_metric_table(aligned)
    summary = summary_metric_table(aligned, monthly)
    return coverage, monthly, summary


def _pred_frame_from_arrays(
    batch: GraphMonthBatch,
    model_name: str,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    mask_np = batch.mask.detach().cpu().numpy().astype(bool)
    y_true = batch.y.detach().cpu().numpy().astype(np.float64)
    stocks = np.asarray(batch.stock_ids, dtype=object)
    return pd.DataFrame(
        {
            "date": np.repeat(batch.date, mask_np.sum()),
            "stock_id": stocks[mask_np],
            "model": model_name,
            "y_true": y_true[mask_np],
            "y_pred": y_pred[mask_np].astype(np.float64),
        }
    )


def run_graph_diagnostic_rerun(project_root: Path, stage5_config_path: Path) -> dict[str, Any]:
    """Refit the main graph model in memory for permutation and static-graph diagnostics."""

    config = load_stage5_yaml_config(stage5_config_path)
    target_col = config.get("data", {}).get("target_col", "target_excess_return")
    panel, _metadata, feature_cols = load_panel_and_metadata(project_root, config)
    manifest = load_graph_manifest(project_root, config)
    manifest_lookup = {pd.Timestamp(row.date): row.pyg_path for row in manifest.itertuples(index=False)}
    panel = panel.loc[panel["date"].isin(manifest_lookup)].copy()
    blocks = make_oos_blocks(panel, config)

    predictions: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    factors: list[pd.DataFrame] = []
    attention: list[pd.DataFrame] = []
    block_runs: list[DiagnosticBlockRun] = []
    block_summaries: list[dict[str, Any]] = []

    for block in blocks:
        train, validation, test = split_block(panel, block)
        train_config = make_train_config(config, block.refit_id)
        model, train_dates, best_loss = fit_graph_model(
            project_root=project_root,
            manifest_lookup=manifest_lookup,
            train=train,
            validation=validation,
            feature_cols=feature_cols,
            target_col=target_col,
            config=train_config,
        )
        device = _device_from_config(train_config.device)
        test_batches = load_graph_batches(project_root, manifest_lookup, test, target_col, device)
        return_attention = train_config.model_type == "gat"
        pred_frame, exp_frame, att_frame = predict_graph_batches(model, test_batches, target_col, return_attention)
        fac_frame = latent_factor_frame(model, train_dates)

        predictions.append(add_block_columns(pred_frame, block, target_col))
        exposures.append(add_block_columns(exp_frame, block, target_col))
        factors.append(add_block_columns(fac_frame, block, target_col))
        if not att_frame.empty:
            attention.append(add_block_columns(att_frame, block, target_col))

        block_runs.append(
            DiagnosticBlockRun(
                refit_id=block.refit_id,
                model=model,
                test_batches=test_batches,
                block=block,
                model_type=train_config.model_type,
                feature_cols=list(feature_cols),
            )
        )
        block_summaries.append(
            {
                "refit_id": block.refit_id,
                "best_validation_loss": float(best_loss),
                "train_config": asdict(train_config),
            }
        )

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    exposure_frame = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    factor_frame = pd.concat(factors, ignore_index=True) if factors else pd.DataFrame()
    attention_frame = pd.concat(attention, ignore_index=True) if attention else pd.DataFrame(columns=["date", "source", "target", "model", "attention_weight"])
    return {
        "config": config,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "predictions": prediction_frame,
        "exposures": exposure_frame,
        "factors": factor_frame,
        "attention": attention_frame,
        "block_runs": block_runs,
        "block_summaries": block_summaries,
        "graph_manifest": manifest,
    }

def permutation_importance_for_graph(
    diagnostic: dict[str, Any],
    candidate_features: list[str],
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    """Compute focused permutation importance for the diagnostic graph rerun."""

    base_predictions = diagnostic["predictions"][["date", "stock_id", "model", "y_true", "y_pred"]].copy()
    _coverage, _monthly, base_summary = summarize_prediction_frame(base_predictions)
    base_row = base_summary.iloc[0]

    feature_to_idx = {feature: idx for idx, feature in enumerate(diagnostic["feature_cols"])}
    rows: list[dict[str, Any]] = []
    for feature in candidate_features:
        if feature not in feature_to_idx:
            continue
        idx = feature_to_idx[feature]
        repeat_metrics: list[dict[str, float]] = []
        for repeat in range(repeats):
            frames: list[pd.DataFrame] = []
            for block_run in diagnostic["block_runs"]:
                model = block_run.model
                model.eval()
                with torch.no_grad():
                    for batch in block_run.test_batches:
                        rng = np.random.default_rng(seed + 1000 * repeat + 10 * block_run.refit_id + idx)
                        perm_idx = torch.as_tensor(rng.permutation(batch.x.shape[0]), device=batch.x.device, dtype=torch.long)
                        x_perm = batch.x.clone()
                        x_perm[:, idx] = x_perm[perm_idx, idx]
                        pred, _beta, _attention = model.predict_with_factor_mean(x_perm, batch.edge_index, batch.edge_weight)
                        frames.append(_pred_frame_from_arrays(batch, model.model_name, pred.detach().cpu().numpy().astype(np.float64)))
            perm_predictions = pd.concat(frames, ignore_index=True)
            _cov, _mon, perm_summary = summarize_prediction_frame(perm_predictions)
            perm_row = perm_summary.iloc[0]
            repeat_metrics.append(
                {
                    "oos_r2": float(perm_row["oos_r2_zero_benchmark"]),
                    "mse": float(perm_row["mse"]),
                    "rank_ic_mean": float(perm_row["rank_ic_mean"]),
                    "cross_sectional_corr_mean": float(perm_row["cross_sectional_corr_mean"]),
                }
            )

        perm_frame = pd.DataFrame(repeat_metrics)
        rows.append(
            {
                "feature": feature,
                "base_oos_r2": float(base_row["oos_r2_zero_benchmark"]),
                "base_mse": float(base_row["mse"]),
                "base_rank_ic_mean": float(base_row["rank_ic_mean"]),
                "base_cross_sectional_corr_mean": float(base_row["cross_sectional_corr_mean"]),
                "permuted_oos_r2": float(perm_frame["oos_r2"].mean()),
                "permuted_mse": float(perm_frame["mse"].mean()),
                "permuted_rank_ic_mean": float(perm_frame["rank_ic_mean"].mean()),
                "permuted_cross_sectional_corr_mean": float(perm_frame["cross_sectional_corr_mean"].mean()),
                "oos_r2_drop": float(base_row["oos_r2_zero_benchmark"] - perm_frame["oos_r2"].mean()),
                "mse_increase": float(perm_frame["mse"].mean() - base_row["mse"]),
                "rank_ic_drop": float(base_row["rank_ic_mean"] - perm_frame["rank_ic_mean"].mean()),
                "cs_corr_drop": float(base_row["cross_sectional_corr_mean"] - perm_frame["cross_sectional_corr_mean"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rank_ic_drop", ascending=False).reset_index(drop=True)


def _batch_with_reference_edges(
    batch: GraphMonthBatch,
    reference_edges: pd.DataFrame,
) -> GraphMonthBatch:
    """Freeze graph edges while keeping current-month node features."""

    node_features = pd.DataFrame(
        batch.x.detach().cpu().numpy().astype(np.float32),
        index=pd.Index(batch.stock_ids, name="stock_id"),
    )
    filtered_edges = reference_edges.loc[
        reference_edges["source"].isin(node_features.index) & reference_edges["target"].isin(node_features.index)
    ].copy()
    if filtered_edges.empty:
        return batch
    pyg = edges_to_pyg_data(filtered_edges, node_features)
    device = batch.x.device
    edge_weight = pyg.edge_weight.to(device) if hasattr(pyg, "edge_weight") else None
    return GraphMonthBatch(
        date=batch.date,
        stock_ids=list(pyg.stock_ids),
        x=pyg.x.to(device),
        edge_index=pyg.edge_index.to(device),
        edge_weight=edge_weight,
        y=batch.y,
        mask=batch.mask,
        date_idx=batch.date_idx,
    )


def frozen_graph_counterfactual(
    project_root: Path,
    diagnostic: dict[str, Any],
    reference_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate the main graph model with frozen test-time edges."""

    manifest = diagnostic["graph_manifest"].copy()
    edge_lookup = load_edge_payload_lookup(project_root, manifest)
    predictions: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []

    for block_run in diagnostic["block_runs"]:
        if reference_mode != "last_validation_end":
            raise ValueError(f"unsupported Stage 8 static graph reference mode: {reference_mode}")
        reference_date = pd.Timestamp(block_run.block.validation_end)
        payload = pd.read_pickle(edge_lookup[reference_date])
        reference_edges = payload["combined_edges"].copy()
        model = block_run.model
        model.eval()
        with torch.no_grad():
            for batch in block_run.test_batches:
                static_batch = _batch_with_reference_edges(batch, reference_edges)
                pred, beta, _attention = model.predict_with_factor_mean(static_batch.x, static_batch.edge_index, static_batch.edge_weight)
                pred_np = pred.detach().cpu().numpy().astype(np.float64)
                beta_np = beta.detach().cpu().numpy().astype(np.float64)
                predictions.append(
                    _pred_frame_from_arrays(static_batch, "graph_conditional_latent_factor_static_test_graph", pred_np)
                )
                mask_np = static_batch.mask.detach().cpu().numpy().astype(bool)
                stocks = np.asarray(static_batch.stock_ids, dtype=object)
                frame = pd.DataFrame(
                    {
                        "date": np.repeat(static_batch.date, mask_np.sum()),
                        "stock_id": stocks[mask_np],
                        "model": "graph_conditional_latent_factor_static_test_graph",
                    }
                )
                for j in range(beta_np.shape[1]):
                    frame[f"beta_{j + 1}"] = beta_np[mask_np, j]
                exposures.append(frame)

    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    exposure_frame = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    return prediction_frame, exposure_frame


def generate_stage4_variant_config(
    project_root: Path,
    base_config_path: Path,
    variant_name: str,
    overrides: dict[str, Any],
    output_root: Path,
) -> Path:
    """Create one generated Stage 4 config for a robustness variant."""

    base = load_stage4_yaml_config(base_config_path)
    config = deep_update(base, overrides)
    graph_dir = output_root / "graphs" / variant_name
    config.setdefault("outputs", {})
    config["outputs"]["graph_dir"] = project_relative_string(project_root, graph_dir)
    config["outputs"]["stats_path"] = project_relative_string(project_root, output_root / "graphs" / f"{variant_name}_stats.csv")
    config["outputs"]["manifest_path"] = project_relative_string(project_root, output_root / "graphs" / f"{variant_name}_manifest.csv")
    config["outputs"]["metadata_path"] = project_relative_string(project_root, output_root / "metadata" / f"{variant_name}_graph_metadata.json")
    config_path = output_root / "generated_configs" / f"{variant_name}_stage4.yaml"
    save_yaml_config(config, config_path)
    return config_path


def generate_stage5_variant_config(
    project_root: Path,
    base_config_path: Path,
    variant_name: str,
    overrides: dict[str, Any],
    manifest_rel_path: str,
    output_root: Path,
) -> Path:
    """Create one generated Stage 5 config for a robustness variant."""

    base = load_stage5_yaml_config(base_config_path)
    config = deep_update(base, overrides)
    config.setdefault("graphs", {})
    config["graphs"]["manifest_path"] = manifest_rel_path
    config.setdefault("outputs", {})
    config["outputs"]["predictions_path"] = project_relative_string(project_root, output_root / "predictions" / f"{variant_name}.pkl")
    config["outputs"]["exposures_path"] = project_relative_string(project_root, output_root / "latent" / f"{variant_name}_exposures.pkl")
    config["outputs"]["factors_path"] = project_relative_string(project_root, output_root / "latent" / f"{variant_name}_factors.pkl")
    config["outputs"]["attention_path"] = project_relative_string(project_root, output_root / "attention" / f"{variant_name}_attention.pkl")
    config["outputs"]["metrics_path"] = project_relative_string(project_root, output_root / "metrics" / f"{variant_name}.csv")
    config["outputs"]["metadata_path"] = project_relative_string(project_root, output_root / "metadata" / f"{variant_name}_model_metadata.json")
    config_path = output_root / "generated_configs" / f"{variant_name}_stage5.yaml"
    save_yaml_config(config, config_path)
    return config_path

def run_graph_robustness_variants(project_root: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build selected graph variants and fit corresponding Stage 5 models."""

    robustness = config.get("robustness", {})
    if not robustness.get("enabled", True):
        return []

    base_stage4 = resolve_project_path(project_root, robustness.get("base_stage4_config_path", "configs/graphs_features500.yaml"))
    base_stage5 = resolve_project_path(project_root, robustness.get("base_stage5_config_path", "configs/graph_model_features500.yaml"))
    output_root = resolve_project_path(project_root, config.get("outputs", {}).get("stage8_root", "outputs/stage8"))
    variants_out: list[dict[str, Any]] = []

    for variant in robustness.get("variants", []):
        name = variant["name"]
        label = variant.get("label", name)
        graph_overrides = variant.get("graph_overrides", {})
        model_overrides = variant.get("model_overrides", {})
        if graph_overrides:
            graph_config_path = generate_stage4_variant_config(project_root, base_stage4, name, graph_overrides, output_root)
            graph_meta = run_stage4_graph_construction(graph_config_path, project_root)
            manifest_rel_path = graph_meta["outputs"]["manifest_path"]
        else:
            base_stage5_cfg = load_stage5_yaml_config(base_stage5)
            manifest_rel_path = str(base_stage5_cfg.get("graphs", {}).get("manifest_path", "outputs/graphs/features500_similarity_hybrid_manifest.csv"))
            graph_config_path = None
            graph_meta = None

        model_config_path = generate_stage5_variant_config(
            project_root=project_root,
            base_config_path=base_stage5,
            variant_name=name,
            overrides=model_overrides,
            manifest_rel_path=manifest_rel_path,
            output_root=output_root,
        )
        model_meta = run_stage5_graph_model(model_config_path, project_root)
        variants_out.append(
            {
                "name": name,
                "label": label,
                "graph_config_path": project_relative_string(project_root, graph_config_path) if graph_config_path is not None else None,
                "graph_metadata": graph_meta,
                "model_config_path": project_relative_string(project_root, model_config_path),
                "model_metadata": model_meta,
                "predictions_path": model_meta["outputs"]["predictions"],
                "attention_path": model_meta["outputs"]["attention"],
            }
        )
    return variants_out


def summarize_attention_variant(
    project_root: Path,
    attention_path: Path,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Attach edge-type metadata to saved GAT attention weights."""

    attention = pd.read_pickle(attention_path).copy()
    if attention.empty:
        return pd.DataFrame()
    attention["date"] = pd.to_datetime(attention["date"])
    edge_lookup = load_edge_payload_lookup(project_root, manifest)
    rows: list[pd.DataFrame] = []
    for date, block in attention.groupby("date", sort=True):
        if pd.Timestamp(date) not in edge_lookup:
            continue
        payload = pd.read_pickle(edge_lookup[pd.Timestamp(date)])
        combined = payload["combined_edges"].copy()
        if combined.empty:
            continue
        combined["pair_key"] = combined.apply(lambda row: "|".join(sorted([str(row["source"]), str(row["target"])])), axis=1)
        month_att = block.copy()
        month_att["pair_key"] = month_att.apply(lambda row: "|".join(sorted([str(row["source"]), str(row["target"])])), axis=1)
        merged = month_att.merge(combined[["pair_key", "edge_types", "edge_type_count", "weight"]], on="pair_key", how="left")
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_portfolio_config_for_models(
    project_root: Path,
    base_stage7_config_path: Path,
    model_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a Stage 7-like config dict for arbitrary model prediction paths."""

    config = load_stage7_config(base_stage7_config_path)
    config["models"] = model_specs
    config.pop("outputs", None)
    return config


def portfolio_summary_for_predictions(
    project_root: Path,
    base_stage7_config_path: Path,
    model_specs: list[dict[str, Any]],
    plot_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the Stage 7 portfolio workflow in memory for selected model predictions."""

    config = build_portfolio_config_for_models(project_root, base_stage7_config_path, model_specs)
    predictions_frames: list[pd.DataFrame] = []
    for spec in model_specs:
        frame = pd.read_pickle(resolve_project_path(project_root, spec["predictions_path"])).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame["model"] = spec["name"]
        predictions_frames.append(frame[["date", "stock_id", "model", "y_true", "y_pred"]].copy())
    predictions = pd.concat(predictions_frames, ignore_index=True)
    aligned, signal_coverage = align_common_signal_panel(predictions)
    signals, merge_stats = merge_signal_inputs(project_root, aligned, config)
    signals["eligible"] = True
    if config.get("portfolio", {}).get("filter_blacklist", True):
        signals.loc[signals["blacklisted_raw_t"], "eligible"] = False
    if config.get("portfolio", {}).get("filter_untradable", True):
        signals.loc[signals["untradable_raw_t"], "eligible"] = False
    weights, strategy_coverage = build_all_weights(signals, config)
    monthly_returns = compute_monthly_portfolio_returns(weights)
    cost_grid = [int(value) for value in config.get("portfolio", {}).get("transaction_cost_bps_grid", [0, 10, 25])]
    net_returns = expand_transaction_cost_scenarios(monthly_returns, cost_grid)
    performance = summarize_performance(net_returns)
    plot_paths: dict[str, str] = {}
    if plot_dir is not None:
        plot_paths = plot_cumulative_returns(net_returns, plot_dir, int(config.get("portfolio", {}).get("main_transaction_cost_bps", 10)))
    return {
        "predictions": predictions,
        "aligned": aligned,
        "signal_coverage": signal_coverage,
        "strategy_coverage": strategy_coverage,
        "weights": weights,
        "monthly_returns": net_returns,
        "performance": performance,
        "merge_stats": merge_stats,
        "plot_paths": plot_paths,
    }


def summarize_robustness_portfolio(performance: pd.DataFrame, main_cost_bps: int) -> pd.DataFrame:
    """Select concise portfolio metrics for graph robustness comparison."""

    main = performance.loc[performance["transaction_cost_bps"] == main_cost_bps].copy()
    keys = [
        ("long_only", "value", "annualized_return", "long_only_value_ann_return"),
        ("long_only", "value", "sharpe_ratio", "long_only_value_sharpe"),
        ("long_short", "equal", "annualized_return", "long_short_equal_ann_return"),
        ("long_short", "equal", "sharpe_ratio", "long_short_equal_sharpe"),
    ]
    rows: list[dict[str, Any]] = []
    for model, group in main.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model}
        for strategy_name, weight_scheme, column, label in keys:
            block = group.loc[(group["strategy_name"] == strategy_name) & (group["weight_scheme"] == weight_scheme)]
            row[label] = float(block.iloc[0][column]) if not block.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def build_main_results_table(stage6: pd.DataFrame, stage7: pd.DataFrame, main_cost_bps: int) -> pd.DataFrame:
    """Combine prediction and portfolio metrics into one report table."""

    stage7_main = stage7.loc[stage7["transaction_cost_bps"] == main_cost_bps].copy()
    picks = [
        ("long_only", "value", "annualized_return", "long_only_value_ann_return"),
        ("long_only", "value", "sharpe_ratio", "long_only_value_sharpe"),
        ("long_short", "equal", "annualized_return", "long_short_equal_ann_return"),
        ("long_short", "equal", "sharpe_ratio", "long_short_equal_sharpe"),
    ]
    rows: list[dict[str, Any]] = []
    for model, group in stage7_main.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model}
        for strategy_name, weight_scheme, column, label in picks:
            block = group.loc[(group["strategy_name"] == strategy_name) & (group["weight_scheme"] == weight_scheme)]
            row[label] = float(block.iloc[0][column]) if not block.empty else np.nan
        rows.append(row)
    portfolio = pd.DataFrame(rows)
    return stage6.merge(portfolio, on="model", how="left")

def plot_stage8_outputs(
    association_top: pd.DataFrame,
    permutation: pd.DataFrame,
    neighbor_monthly: pd.DataFrame,
    neighbor_edge_mix: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    robustness_portfolio: pd.DataFrame,
    attention_summary: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Create report-ready Stage 8 figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths: dict[str, str] = {}

    if not association_top.empty:
        pivot = association_top.pivot(index="feature", columns="model", values="max_abs_spearman").fillna(0.0)
        top_features = pivot.max(axis=1).sort_values(ascending=False).head(min(15, len(pivot))).index
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot.loc[top_features], cmap="YlGnBu", annot=True, fmt=".2f", ax=ax)
        ax.set_title("Top Feature-to-Exposure Links")
        plt.tight_layout()
        path = output_dir / "stage8_feature_exposure_heatmap.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["feature_exposure_heatmap"] = str(path)

    if not permutation.empty:
        top_perm = permutation.sort_values("rank_ic_drop", ascending=False).head(min(15, len(permutation)))
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=top_perm, x="rank_ic_drop", y="feature", hue="feature", dodge=False, legend=False, ax=ax, palette="deep")
        ax.set_title("Graph Model Permutation Importance")
        ax.set_xlabel("Rank IC Drop")
        ax.set_ylabel("")
        plt.tight_layout()
        path = output_dir / "stage8_permutation_importance.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["permutation_importance"] = str(path)

    if not neighbor_monthly.empty:
        summary = neighbor_monthly[["mean_degree_universe", "mean_degree_top", "top_edge_density_share"]].mean().rename_axis("metric").reset_index(name="value")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=summary, x="metric", y="value", hue="metric", dodge=False, legend=False, ax=ax, palette="muted")
        ax.set_title("Graph Neighborhood Summary for Top-Decile Picks")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        path = output_dir / "stage8_neighbor_summary.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["neighbor_summary"] = str(path)

    if not neighbor_edge_mix.empty:
        mix = neighbor_edge_mix.groupby("edge_type", as_index=False)["share"].mean().sort_values("share", ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=mix, x="edge_type", y="share", hue="edge_type", dodge=False, legend=False, ax=ax, palette="deep")
        ax.set_title("Edge-Type Mix Around Top-Decile Graph Picks")
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        path = output_dir / "stage8_neighbor_edge_mix.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["neighbor_edge_mix"] = str(path)

    if not robustness_summary.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=robustness_summary, x="model", y="rank_ic_mean", hue="model", dodge=False, legend=False, ax=axes[0], palette="deep")
        axes[0].set_title("Graph Variant Rank IC")
        axes[0].tick_params(axis="x", rotation=20)
        sns.barplot(data=robustness_summary, x="model", y="cross_sectional_corr_mean", hue="model", dodge=False, legend=False, ax=axes[1], palette="deep")
        axes[1].set_title("Graph Variant Cross-Sectional Corr")
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        path = output_dir / "stage8_robustness_prediction_bars.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["robustness_prediction_bars"] = str(path)

    if not robustness_portfolio.empty:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(data=robustness_portfolio, x="model", y="long_only_value_sharpe", hue="model", dodge=False, legend=False, ax=axes[0], palette="deep")
        axes[0].set_title("Graph Variant Long-Only Value Sharpe")
        axes[0].tick_params(axis="x", rotation=20)
        sns.barplot(data=robustness_portfolio, x="model", y="long_short_equal_sharpe", hue="model", dodge=False, legend=False, ax=axes[1], palette="deep")
        axes[1].set_title("Graph Variant Long-Short Equal Sharpe")
        axes[1].tick_params(axis="x", rotation=20)
        plt.tight_layout()
        path = output_dir / "stage8_robustness_portfolio_bars.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["robustness_portfolio_bars"] = str(path)

    if not attention_summary.empty:
        top = attention_summary.groupby("edge_types", as_index=False)["attention_weight"].mean().sort_values("attention_weight", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=top, x="edge_types", y="attention_weight", hue="edge_types", dodge=False, legend=False, ax=ax, palette="deep")
        ax.set_title("GAT Attention by Edge-Type Combination")
        ax.tick_params(axis="x", rotation=25)
        plt.tight_layout()
        path = output_dir / "stage8_gat_attention_edge_types.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths["gat_attention_edge_types"] = str(path)

    return plot_paths


def write_stage8_summary(
    project_root: Path,
    report_path: Path,
    stage8_inputs: dict[str, Any],
    association_top: pd.DataFrame,
    permutation: pd.DataFrame,
    neighbor_monthly: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    robustness_portfolio: pd.DataFrame,
    attention_summary: pd.DataFrame,
    plot_paths: dict[str, str],
    variant_results: list[dict[str, Any]],
) -> None:
    """Write the final consolidated Stage 8 markdown summary."""

    stage6 = stage8_inputs["stage6_summary"].copy()
    stage7 = stage8_inputs["stage7_performance"].copy()
    graph_row = stage6.loc[stage6["model"] == "graph_conditional_latent_factor"].iloc[0]
    mlp_row = stage6.loc[stage6["model"] == "mlp_predictor"].iloc[0]
    ipca_row = stage6.loc[stage6["model"] == "ipca_style"].iloc[0]
    cae_row = stage6.loc[stage6["model"] == "conditional_autoencoder_style"].iloc[0]

    main_cost = 10
    long_only = stage7.loc[
        (stage7["transaction_cost_bps"] == main_cost)
        & (stage7["strategy_name"] == "long_only")
        & (stage7["weight_scheme"] == "value")
    ].copy()
    long_short = stage7.loc[
        (stage7["transaction_cost_bps"] == main_cost)
        & (stage7["strategy_name"] == "long_short")
        & (stage7["weight_scheme"] == "equal")
    ].copy()
    graph_lo = long_only.loc[long_only["model"] == "graph_conditional_latent_factor"].iloc[0]
    graph_ls = long_short.loc[long_short["model"] == "graph_conditional_latent_factor"].iloc[0]

    strongest_perm = permutation.head(5)[["feature", "rank_ic_drop"]] if not permutation.empty else pd.DataFrame()
    top_links = association_top.groupby("model").head(3) if not association_top.empty else pd.DataFrame()
    neighbor_avg = neighbor_monthly.mean(numeric_only=True) if not neighbor_monthly.empty else pd.Series(dtype="float64")
    has_attention = not attention_summary.empty

    text = f"""# Stage 8 Final Summary

## Main Finding

The core research question is whether adding graph structure to a conditional latent factor model improves dynamic exposure estimation enough to beat characteristic-only models in out-of-sample pricing, prediction, and portfolio performance.

On the current default `features500` main specification, the strongest evidence is that graph structure adds **incremental cross-sectional structure and economic value**, but not yet a clean all-metric victory.

- Stage 6 graph-model rank IC: {float(graph_row['rank_ic_mean']):.4f}
- Stage 6 graph-model cross-sectional correlation: {float(graph_row['cross_sectional_corr_mean']):.4f}
- Stage 6 graph-model OOS R^2: {float(graph_row['oos_r2_zero_benchmark']):.4f}
- Stage 7 graph-model long-only value-weight annualized return at 10 bps: {float(graph_lo['annualized_return']):.4f}
- Stage 7 graph-model long-only value-weight Sharpe at 10 bps: {float(graph_lo['sharpe_ratio']):.4f}
- Stage 7 graph-model long-short equal-weight annualized return at 10 bps: {float(graph_ls['annualized_return']):.4f}
- Stage 7 graph-model long-short equal-weight Sharpe at 10 bps: {float(graph_ls['sharpe_ratio']):.4f}

## Benchmark Comparison

Relative to the characteristic-only nonlinear latent benchmark (`conditional_autoencoder_style`), the graph model improves rank IC ({float(graph_row['rank_ic_mean']):.4f} vs {float(cae_row['rank_ic_mean']):.4f}) and cross-sectional correlation ({float(graph_row['cross_sectional_corr_mean']):.4f} vs {float(cae_row['cross_sectional_corr_mean']):.4f}).

Relative to `ipca_style`, the graph model slightly improves ranking ({float(graph_row['rank_ic_mean']):.4f} vs {float(ipca_row['rank_ic_mean']):.4f}) but not OOS R^2 ({float(graph_row['oos_r2_zero_benchmark']):.4f} vs {float(ipca_row['oos_r2_zero_benchmark']):.4f}).

Relative to the direct `mlp_predictor`, the graph model has weaker rank IC ({float(graph_row['rank_ic_mean']):.4f} vs {float(mlp_row['rank_ic_mean']):.4f}) but materially better OOS R^2 ({float(graph_row['oos_r2_zero_benchmark']):.4f} vs {float(mlp_row['oos_r2_zero_benchmark']):.4f}) and a much stronger asset-pricing interpretation.

## Economic Value

The clearest economic win is in the long-only tests. The current graph model produces the best long-only Sharpe and return among the four implemented main-spec models. In long-short portfolios, the graph model is competitive on return, but the IPCA-style benchmark remains hard to beat on Sharpe.

That pattern suggests the graph-enhanced beta function is helping identify better top-ranked names for implementable portfolios, while the risk control side of the long-short book still has room to improve.

## Interpretation

The interpretation work in this stage stays close to the models we actually implemented.

"""
    if not top_links.empty:
        text += "Top feature-to-exposure links from the saved latent outputs:\n\n"
        for row in top_links.itertuples(index=False):
            text += f"- `{row.model}`: `{row.feature}` (avg abs monthly Spearman {row.max_abs_spearman:.3f})\n"
        text += "\n"
    if not strongest_perm.empty:
        text += "Focused graph-model permutation importance from the Stage 8 diagnostic rerun:\n\n"
        for row in strongest_perm.itertuples(index=False):
            text += f"- `{row.feature}`: rank IC drop {row.rank_ic_drop:.4f}\n"
        text += "\n"
    if not neighbor_avg.empty:
        text += (
            "Graph-neighborhood summaries show that top-decile graph picks sit in denser parts of the monthly graph than the average stock "
            f"(mean degree top {neighbor_avg.get('mean_degree_top', np.nan):.2f} vs universe {neighbor_avg.get('mean_degree_universe', np.nan):.2f}).\n\n"
        )
    if has_attention:
        text += "A GAT robustness run was also completed, so attention-weight summaries are available as exploratory evidence.\n\n"
    else:
        text += "The default Stage 5 model is GCN, so attention evidence is only available if the exploratory GAT robustness run succeeds.\n\n"

    text += "## Robustness\n\n"
    if not robustness_summary.empty:
        text += "Focused graph robustness checks were run on a small number of variants that stay close to the implemented pipeline.\n\n"
        for row in robustness_summary.sort_values("rank_ic_mean", ascending=False).itertuples(index=False):
            text += (
                f"- `{row.model}`: OOS R^2 {row.oos_r2_zero_benchmark:.4f}, rank IC {row.rank_ic_mean:.4f}, "
                f"CS corr {row.cross_sectional_corr_mean:.4f}, pricing RMSE {row.pricing_error_monthly_rmse:.4f}\n"
            )
        text += "\n"
    if not robustness_portfolio.empty:
        text += "Selected portfolio robustness diagnostics:\n\n"
        for row in robustness_portfolio.sort_values("long_only_value_sharpe", ascending=False).itertuples(index=False):
            text += (
                f"- `{row.model}`: long-only value Sharpe {row.long_only_value_sharpe:.4f}, "
                f"long-short equal Sharpe {row.long_short_equal_sharpe:.4f}\n"
            )
        text += "\n"
    if variant_results:
        text += "Implemented robustness variants:\n\n"
        for variant in variant_results:
            text += f"- `{variant['name']}`: {variant['label']}\n"
        text += "\n"

    text += """## Strong Evidence

1. Graph structure adds incremental value beyond the characteristic-only CAE-style latent benchmark on ranking-style prediction metrics.
2. The graph model delivers the strongest long-only main-spec portfolio outcome among the implemented models.
3. The project’s main takeaway is more convincing on **ranking and economic value** than on raw OOS R^2.

## Tentative Evidence

1. Pricing gains are modest and not yet decisive.
2. The long-short edge is mixed because IPCA-style still competes strongly on Sharpe.
3. Any attention-based interpretation is exploratory, because the default main result is still a GCN run.
4. Robustness beyond the `features500` universe remains incomplete because a broader `features/` panel was not rebuilt in this stage.

## Limitations

1. The default OOS window is still short.
2. The graph model remains a compact first-pass latent-factor system, not a no-arbitrage structural model.
3. Factor forecasting still uses the train-window mean latent premium.
4. The stored data does not provide point-in-time industry labels, so the graph remains similarity-based.
5. Stage 8 interpretation for the graph model relies partly on a diagnostic rerun because the earlier stages did not save model checkpoints.

## Next-Step Extensions

1. Save full per-refit checkpoints by default so interpretation no longer needs diagnostic reruns.
2. Add a stronger no-arbitrage loss or SDF-style constraint.
3. Replace the homogeneous graph with a multi-relation graph encoder that preserves edge types explicitly.
4. Extend the full pipeline to the broader `features/` universe.
5. Add longer OOS windows and more systematic hyperparameter sweeps.

## Figures

"""
    for key, path in plot_paths.items():
        text += f"- `{project_relative_string(project_root, path)}`\n"

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")

def run_stage8_analysis(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run the full Stage 8 interpretability and robustness workflow."""

    started_at = time.time()
    config = load_yaml(config_path)
    outputs_cfg = config.get("outputs", {})
    stage8_root = resolve_project_path(project_root, outputs_cfg.get("stage8_root", "outputs/stage8"))
    table_dir = resolve_project_path(project_root, outputs_cfg.get("table_dir", "outputs/stage8/tables"))
    plot_dir = resolve_project_path(project_root, outputs_cfg.get("plot_dir", "outputs/stage8/plots"))
    metadata_path = resolve_project_path(project_root, outputs_cfg.get("metadata_path", "outputs/metadata/stage8_analysis_metadata.json"))
    report_path = resolve_project_path(project_root, outputs_cfg.get("report_path", "reports/stage8_final_summary.md"))
    for directory in [stage8_root, table_dir, plot_dir, metadata_path.parent, report_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    stage8_inputs = load_stage8_inputs(project_root, config)
    association = feature_exposure_association(
        panel=stage8_inputs["panel"],
        exposures=stage8_inputs["main_exposures"],
        feature_cols=stage8_inputs["feature_cols"],
        model_names=["ipca_style", "conditional_autoencoder_style", "graph_conditional_latent_factor"],
    )
    association_top = summarize_top_feature_links(association, top_n=int(config.get("interpretability", {}).get("top_feature_links_per_model", 10)))

    graph_main_predictions = stage8_inputs["stage5_predictions"].loc[
        stage8_inputs["stage5_predictions"]["model"] == "graph_conditional_latent_factor",
        ["date", "stock_id", "model", "y_true", "y_pred"],
    ].copy()
    neighbor_monthly, neighbor_edge_mix = graph_neighbor_summary(
        project_root=project_root,
        manifest=stage8_inputs["graph_manifest"],
        graph_predictions=graph_main_predictions,
        top_quantile=float(config.get("interpretability", {}).get("top_quantile", 0.1)),
    )

    diagnostic_cfg_path = resolve_project_path(
        project_root,
        config.get("diagnostic_graph", {}).get("stage5_config_path", "configs/graph_model_features500.yaml"),
    )
    diagnostic = run_graph_diagnostic_rerun(project_root, diagnostic_cfg_path)
    diagnostic_predictions_path = stage8_root / "predictions" / "graph_main_diagnostic.pkl"
    diagnostic_exposures_path = stage8_root / "latent" / "graph_main_diagnostic_exposures.pkl"
    diagnostic_factors_path = stage8_root / "latent" / "graph_main_diagnostic_factors.pkl"
    diagnostic_attention_path = stage8_root / "attention" / "graph_main_diagnostic_attention.pkl"
    for path in [diagnostic_predictions_path, diagnostic_exposures_path, diagnostic_factors_path, diagnostic_attention_path]:
        _ensure_parent(path)
    diagnostic["predictions"].to_pickle(diagnostic_predictions_path)
    diagnostic["exposures"].to_pickle(diagnostic_exposures_path)
    diagnostic["factors"].to_pickle(diagnostic_factors_path)
    diagnostic["attention"].to_pickle(diagnostic_attention_path)

    candidate_features = select_permutation_candidates(
        association=association,
        feature_cols=stage8_inputs["feature_cols"],
        model_name="graph_conditional_latent_factor",
        top_n=int(config.get("interpretability", {}).get("permutation_candidate_features", 25)),
    )
    permutation = permutation_importance_for_graph(
        diagnostic=diagnostic,
        candidate_features=candidate_features,
        repeats=int(config.get("interpretability", {}).get("permutation_repeats", 1)),
        seed=int(config.get("interpretability", {}).get("permutation_seed", 20260409)),
    )

    static_predictions, static_exposures = frozen_graph_counterfactual(
        project_root=project_root,
        diagnostic=diagnostic,
        reference_mode=str(config.get("diagnostic_graph", {}).get("static_reference_mode", "last_validation_end")),
    )
    static_predictions_path = stage8_root / "predictions" / "graph_static_test_graph.pkl"
    static_exposures_path = stage8_root / "latent" / "graph_static_test_graph_exposures.pkl"
    static_predictions.to_pickle(static_predictions_path)
    static_exposures.to_pickle(static_exposures_path)

    variant_results = run_graph_robustness_variants(project_root, config)
    variant_model_specs = [
        {"name": "graph_main_diagnostic", "predictions_path": project_relative_string(project_root, diagnostic_predictions_path)},
        {"name": "graph_conditional_latent_factor_static_test_graph", "predictions_path": project_relative_string(project_root, static_predictions_path)},
    ]
    for variant in variant_results:
        variant_model_specs.append({"name": variant["name"], "predictions_path": variant["predictions_path"]})

    robustness_summary = pd.DataFrame()
    robustness_portfolio = pd.DataFrame()
    attention_summary = pd.DataFrame()
    stage7_base_config = resolve_project_path(project_root, config.get("portfolio", {}).get("base_stage7_config_path", "configs/portfolio_features500.yaml"))
    if variant_model_specs:
        robustness_frames: list[pd.DataFrame] = []
        for spec in variant_model_specs:
            frame = pd.read_pickle(resolve_project_path(project_root, spec["predictions_path"])).copy()
            frame["date"] = pd.to_datetime(frame["date"])
            frame["model"] = spec["name"]
            robustness_frames.append(frame[["date", "stock_id", "model", "y_true", "y_pred"]].copy())
        robustness_predictions = pd.concat(robustness_frames, ignore_index=True)
        _rob_cov, _rob_monthly, robustness_summary = summarize_prediction_frame(robustness_predictions)

        robustness_port = portfolio_summary_for_predictions(
            project_root=project_root,
            base_stage7_config_path=stage7_base_config,
            model_specs=variant_model_specs,
            plot_dir=stage8_root / "plots" / "portfolio_robustness",
        )
        robustness_portfolio = summarize_robustness_portfolio(
            performance=robustness_port["performance"],
            main_cost_bps=int(config.get("portfolio", {}).get("main_transaction_cost_bps", 10)),
        )

    for variant in variant_results:
        if "gat" in variant["name"]:
            attention_path = resolve_project_path(project_root, variant["attention_path"])
            manifest = pd.read_csv(resolve_project_path(project_root, variant["model_metadata"]["graph_manifest"]))
            manifest["date"] = pd.to_datetime(manifest["date"])
            attention_summary = summarize_attention_variant(project_root, attention_path, manifest)
            break

    main_results_table = build_main_results_table(
        stage6=stage8_inputs["stage6_summary"],
        stage7=stage8_inputs["stage7_performance"],
        main_cost_bps=int(config.get("portfolio", {}).get("main_transaction_cost_bps", 10)),
    )
    main_results_table_path = table_dir / "stage8_main_results_table.csv"
    association_path = table_dir / "stage8_feature_exposure_association.csv"
    association_top_path = table_dir / "stage8_feature_exposure_top_links.csv"
    permutation_path = table_dir / "stage8_permutation_importance.csv"
    neighbor_monthly_path = table_dir / "stage8_neighbor_monthly_summary.csv"
    neighbor_edge_mix_path = table_dir / "stage8_neighbor_edge_mix.csv"
    robustness_summary_path = table_dir / "stage8_graph_robustness_summary.csv"
    robustness_portfolio_path = table_dir / "stage8_graph_robustness_portfolio.csv"
    attention_summary_path = table_dir / "stage8_gat_attention_summary.csv"
    for path in [
        main_results_table_path,
        association_path,
        association_top_path,
        permutation_path,
        neighbor_monthly_path,
        neighbor_edge_mix_path,
        robustness_summary_path,
        robustness_portfolio_path,
        attention_summary_path,
    ]:
        _ensure_parent(path)

    main_results_table.to_csv(main_results_table_path, index=False)
    association.to_csv(association_path, index=False)
    association_top.to_csv(association_top_path, index=False)
    permutation.to_csv(permutation_path, index=False)
    neighbor_monthly.to_csv(neighbor_monthly_path, index=False)
    neighbor_edge_mix.to_csv(neighbor_edge_mix_path, index=False)
    robustness_summary.to_csv(robustness_summary_path, index=False)
    robustness_portfolio.to_csv(robustness_portfolio_path, index=False)
    attention_summary.to_csv(attention_summary_path, index=False)

    plot_paths = plot_stage8_outputs(
        association_top=association_top,
        permutation=permutation,
        neighbor_monthly=neighbor_monthly,
        neighbor_edge_mix=neighbor_edge_mix,
        robustness_summary=robustness_summary,
        robustness_portfolio=robustness_portfolio,
        attention_summary=attention_summary,
        output_dir=plot_dir,
    )
    write_stage8_summary(
        project_root=project_root,
        report_path=report_path,
        stage8_inputs=stage8_inputs,
        association_top=association_top,
        permutation=permutation,
        neighbor_monthly=neighbor_monthly,
        robustness_summary=robustness_summary,
        robustness_portfolio=robustness_portfolio,
        attention_summary=attention_summary,
        plot_paths=plot_paths,
        variant_results=variant_results,
    )

    metadata = {
        "stage": "stage8_interpretability_robustness",
        "config_path": project_relative_string(project_root, config_path),
        "diagnostic_graph_config_path": project_relative_string(project_root, diagnostic_cfg_path),
        "candidate_features": candidate_features,
        "variant_results": variant_results,
        "diagnostic_block_summaries": diagnostic["block_summaries"],
        "outputs": {
            "stage8_root": project_relative_string(project_root, stage8_root),
            "tables": {
                "main_results": project_relative_string(project_root, main_results_table_path),
                "association": project_relative_string(project_root, association_path),
                "association_top": project_relative_string(project_root, association_top_path),
                "permutation": project_relative_string(project_root, permutation_path),
                "neighbor_monthly": project_relative_string(project_root, neighbor_monthly_path),
                "neighbor_edge_mix": project_relative_string(project_root, neighbor_edge_mix_path),
                "robustness_summary": project_relative_string(project_root, robustness_summary_path),
                "robustness_portfolio": project_relative_string(project_root, robustness_portfolio_path),
                "attention_summary": project_relative_string(project_root, attention_summary_path),
            },
            "plots": {key: project_relative_string(project_root, path) for key, path in plot_paths.items()},
            "report": project_relative_string(project_root, report_path),
            "diagnostic_predictions": project_relative_string(project_root, diagnostic_predictions_path),
            "diagnostic_exposures": project_relative_string(project_root, diagnostic_exposures_path),
            "diagnostic_factors": project_relative_string(project_root, diagnostic_factors_path),
            "static_predictions": project_relative_string(project_root, static_predictions_path),
        },
        "main_stage6_summary_path": config.get("main_results", {}).get("stage6_summary_path"),
        "main_stage7_performance_path": config.get("main_results", {}).get("stage7_performance_path"),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print("Saved Stage 8 outputs:")
    print(f"  tables:   {table_dir}")
    print(f"  plots:    {plot_dir}")
    print(f"  report:   {report_path}")
    print(f"  metadata: {metadata_path}")
    if not robustness_summary.empty:
        print(robustness_summary.to_string(index=False))
    return metadata
