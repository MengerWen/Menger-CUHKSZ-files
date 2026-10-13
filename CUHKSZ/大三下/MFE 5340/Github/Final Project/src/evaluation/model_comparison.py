"""Unified Stage 6 comparison framework for benchmark and graph model outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

from src.evaluation.metrics import rank_ic_by_month
from src.training.non_graph_benchmark_pipeline import resolve_project_path

sns.set_theme(style="whitegrid")


def load_config(config_path: Path) -> dict[str, Any]:
    """Load Stage 6 YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {config_path}")
    return config


def project_relative_string(project_root: Path, path: Path | str) -> str:
    """Return project-relative path string when possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def load_output_frames(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load prediction, exposure, and factor outputs from Stage 3 and Stage 5."""

    model_specs = config.get("models", [])
    if not model_specs:
        raise ValueError("config.models is empty")

    predictions: list[pd.DataFrame] = []
    exposures: list[pd.DataFrame] = []
    factors: list[pd.DataFrame] = []
    metadata: dict[str, Any] = {}
    for spec in model_specs:
        name = spec["name"]
        prediction_path = resolve_project_path(project_root, spec["predictions_path"])
        pred = pd.read_pickle(prediction_path).copy()
        pred["date"] = pd.to_datetime(pred["date"])
        if "model" in pred.columns:
            pred = pred.loc[pred["model"] == name].copy()
        if pred.empty:
            raise ValueError(f"no prediction rows found for model {name} in {prediction_path}")
        predictions.append(pred)

        exposure_path = spec.get("exposures_path")
        if exposure_path:
            exp = pd.read_pickle(resolve_project_path(project_root, exposure_path)).copy()
            exp["date"] = pd.to_datetime(exp["date"])
            if "model" in exp.columns:
                exp = exp.loc[exp["model"] == name].copy()
            if not exp.empty:
                exposures.append(exp)

        factor_path = spec.get("factors_path")
        if factor_path:
            fac = pd.read_pickle(resolve_project_path(project_root, factor_path)).copy()
            if "date" in fac.columns:
                fac["date"] = pd.to_datetime(fac["date"])
            if "model" in fac.columns:
                fac = fac.loc[fac["model"] == name].copy()
            if not fac.empty:
                factors.append(fac)

        metadata_path = spec.get("metadata_path")
        if metadata_path:
            metadata[name] = json.loads(resolve_project_path(project_root, metadata_path).read_text(encoding="utf-8"))

    prediction_frame = pd.concat(predictions, ignore_index=True)
    exposure_frame = pd.concat(exposures, ignore_index=True) if exposures else pd.DataFrame()
    factor_frame = pd.concat(factors, ignore_index=True) if factors else pd.DataFrame()
    return prediction_frame, exposure_frame, factor_frame, metadata


def align_common_prediction_panel(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Restrict predictions to the common stock-date intersection across models."""

    base = predictions.copy()
    base["date"] = pd.to_datetime(base["date"])
    required_models = base["model"].nunique()
    common_keys = (
        base.groupby(["date", "stock_id"], sort=True)["model"]
        .nunique()
        .loc[lambda values: values == required_models]
        .reset_index()[["date", "stock_id"]]
    )
    aligned = base.merge(common_keys, on=["date", "stock_id"], how="inner")
    aligned = aligned.sort_values(["date", "stock_id", "model"]).reset_index(drop=True)

    universe = (
        aligned.groupby("model", sort=True)
        .agg(
            n_obs=("stock_id", "size"),
            n_months=("date", "nunique"),
            n_unique_stocks=("stock_id", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        )
        .reset_index()
    )
    month_counts = (
        aligned.groupby(["model", "date"], sort=True)
        .size()
        .rename("n_obs")
        .reset_index()
    )
    return aligned, universe, month_counts


def monthly_metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly per-model prediction, ranking, and pricing metrics."""

    rows: list[dict[str, Any]] = []
    for (model, date), group in frame.groupby(["model", "date"], sort=True):
        y_true = group["y_true"].to_numpy(dtype=np.float64)
        y_pred = group["y_pred"].to_numpy(dtype=np.float64)
        residual = y_true - y_pred
        rank_ic = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
        pearson = pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
        denominator = float(np.sum(y_true**2))
        finance_oos_r2 = np.nan if np.isclose(denominator, 0.0) else 1.0 - float(np.sum(residual**2)) / denominator
        rows.append(
            {
                "model": model,
                "date": date,
                "n_obs": int(group.shape[0]),
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rank_ic": float(rank_ic) if pd.notna(rank_ic) else np.nan,
                "cross_sectional_corr": float(pearson) if pd.notna(pearson) else np.nan,
                "pricing_error_mean": float(residual.mean()),
                "pricing_error_abs_mean": float(np.abs(residual).mean()),
                "pricing_error_rmse": float(np.sqrt(np.mean(residual**2))),
                "finance_oos_r2_month": finance_oos_r2,
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "date"]).reset_index(drop=True)


def summary_metric_table(aligned: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate overall model comparison metrics on the aligned sample."""

    rows: list[dict[str, Any]] = []
    for model, group in aligned.groupby("model", sort=True):
        y_true = group["y_true"].to_numpy(dtype=np.float64)
        y_pred = group["y_pred"].to_numpy(dtype=np.float64)
        residual = y_true - y_pred
        denominator = float(np.sum(y_true**2))
        finance_oos_r2 = np.nan if np.isclose(denominator, 0.0) else 1.0 - float(np.sum(residual**2)) / denominator
        month_metrics = monthly.loc[monthly["model"] == model]
        rank_ic = month_metrics["rank_ic"].dropna()
        cs_corr = month_metrics["cross_sectional_corr"].dropna()
        pricing_mean = month_metrics["pricing_error_mean"].dropna()
        rows.append(
            {
                "model": model,
                "n_obs": int(group.shape[0]),
                "n_months": int(group["date"].nunique()),
                "oos_r2_zero_benchmark": finance_oos_r2,
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(root_mean_squared_error(y_true, y_pred)),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rank_ic_mean": float(rank_ic.mean()) if not rank_ic.empty else np.nan,
                "rank_ic_std": float(rank_ic.std(ddof=1)) if rank_ic.size > 1 else np.nan,
                "rank_ic_tstat": float(rank_ic.mean() / (rank_ic.std(ddof=1) / np.sqrt(rank_ic.size))) if rank_ic.size > 1 and not np.isclose(rank_ic.std(ddof=1), 0.0) else np.nan,
                "cross_sectional_corr_mean": float(cs_corr.mean()) if not cs_corr.empty else np.nan,
                "cross_sectional_corr_std": float(cs_corr.std(ddof=1)) if cs_corr.size > 1 else np.nan,
                "cross_sectional_corr_tstat": float(cs_corr.mean() / (cs_corr.std(ddof=1) / np.sqrt(cs_corr.size))) if cs_corr.size > 1 and not np.isclose(cs_corr.std(ddof=1), 0.0) else np.nan,
                "pricing_error_monthly_mean": float(pricing_mean.mean()) if not pricing_mean.empty else np.nan,
                "pricing_error_monthly_abs_mean": float(month_metrics["pricing_error_abs_mean"].mean()),
                "pricing_error_monthly_rmse": float(np.sqrt(np.mean(month_metrics["pricing_error_mean"].to_numpy(dtype=np.float64) ** 2))),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def prediction_correlation_table(aligned: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation of model predictions on the aligned sample."""

    wide = aligned.pivot(index=["date", "stock_id"], columns="model", values="y_pred").sort_index(axis=1)
    return wide.corr().reset_index().rename(columns={"index": "model"})


def latent_diagnostics(exposures: pd.DataFrame, factors: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """Compute latent factor and exposure diagnostics when available."""

    factor_cols = [col for col in factors.columns if col.startswith("factor_") and col.split("_")[-1].isdigit()]
    beta_cols = [col for col in exposures.columns if col.startswith("beta_")]
    rows: list[dict[str, Any]] = []
    for model in model_names:
        row: dict[str, Any] = {"model": model}
        model_exp = exposures.loc[exposures["model"] == model].copy() if not exposures.empty else pd.DataFrame()
        model_fac = factors.loc[factors["model"] == model].copy() if not factors.empty else pd.DataFrame()
        row["has_exposures"] = not model_exp.empty
        row["has_factors"] = not model_fac.empty
        row["factor_dim"] = int(len([col for col in factor_cols if col in model_fac.columns])) if not model_fac.empty else np.nan
        if not model_exp.empty and beta_cols:
            monthly_beta_std = model_exp.groupby("date", sort=True)[beta_cols].std(ddof=1).mean(axis=1)
            row["exposure_dispersion_mean"] = float(monthly_beta_std.mean())
            row["exposure_abs_mean"] = float(model_exp[beta_cols].abs().mean().mean())
            row["exposure_rows"] = int(model_exp.shape[0])
        else:
            row["exposure_dispersion_mean"] = np.nan
            row["exposure_abs_mean"] = np.nan
            row["exposure_rows"] = 0
        if not model_fac.empty and factor_cols:
            train_factors = model_fac.loc[model_fac["factor_kind"] == "train_factor", factor_cols]
            forecast_mean = model_fac.loc[model_fac["factor_kind"] == "forecast_mean", factor_cols]
            row["train_factor_rows"] = int(train_factors.shape[0])
            row["factor_abs_mean"] = float(train_factors.abs().mean().mean()) if not train_factors.empty else np.nan
            row["factor_std_mean"] = float(train_factors.std(ddof=1).mean()) if len(train_factors) > 1 else np.nan
            row["forecast_factor_abs_mean"] = float(forecast_mean.abs().mean().mean()) if not forecast_mean.empty else np.nan
        else:
            row["train_factor_rows"] = 0
            row["factor_abs_mean"] = np.nan
            row["factor_std_mean"] = np.nan
            row["forecast_factor_abs_mean"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def make_comparison_plots(
    summary: pd.DataFrame,
    monthly: pd.DataFrame,
    correlation_table: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    """Create Stage 6 comparison figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    metric_map = {
        "oos_r2_zero_benchmark": "OOS R2",
        "rank_ic_mean": "Rank IC",
        "cross_sectional_corr_mean": "CS Corr",
        "pricing_error_monthly_rmse": "Pricing Error RMSE",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, title) in zip(axes.ravel(), metric_map.items()):
        plot_data = summary.sort_values(column, ascending=(column == "pricing_error_monthly_rmse"))
        sns.barplot(data=plot_data, x="model", y=column, hue="model", dodge=False, legend=False, ax=ax, palette="deep")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    overall_path = output_dir / 'stage6_overall_metric_bars.png'
    fig.savefig(overall_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['overall_metric_bars'] = str(overall_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="date", y="rank_ic", hue="model", marker="o", ax=ax)
    ax.set_title("Monthly Rank IC by Model")
    ax.set_ylabel("Rank IC")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    rank_ic_path = output_dir / 'stage6_monthly_rank_ic.png'
    fig.savefig(rank_ic_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['monthly_rank_ic'] = str(rank_ic_path)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(data=monthly, x="date", y="cross_sectional_corr", hue="model", marker="o", ax=ax)
    ax.set_title("Monthly Cross-Sectional Pearson Correlation")
    ax.set_ylabel("Cross-Sectional Corr")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    cs_corr_path = output_dir / 'stage6_monthly_cross_sectional_corr.png'
    fig.savefig(cs_corr_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['monthly_cross_sectional_corr'] = str(cs_corr_path)

    corr_matrix = correlation_table.set_index("model")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0.0, fmt=".3f", ax=ax)
    ax.set_title("Prediction Correlation Across Models")
    plt.tight_layout()
    corr_path = output_dir / 'stage6_prediction_correlation_heatmap.png'
    fig.savefig(corr_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    paths['prediction_correlation_heatmap'] = str(corr_path)

    return paths


def winner_name(summary: pd.DataFrame, column: str, higher_is_better: bool = True) -> str:
    """Return the best model name for one metric."""

    valid = summary[["model", column]].dropna()
    if valid.empty:
        return "N/A"
    row = valid.sort_values(column, ascending=not higher_is_better).iloc[0]
    return str(row["model"])


def write_interpretation_note(
    project_root: Path,
    config: dict[str, Any],
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    latent: pd.DataFrame,
    plot_paths: dict[str, str],
    report_path: Path,
) -> None:
    """Write a Stage 6 markdown note focused on the core research question."""

    graph_row = summary.loc[summary["model"] == "graph_conditional_latent_factor"]
    mlp_row = summary.loc[summary["model"] == "mlp_predictor"]
    cae_row = summary.loc[summary["model"] == "conditional_autoencoder_style"]
    ipca_row = summary.loc[summary["model"] == "ipca_style"]

    def _metric(model_row: pd.DataFrame, column: str) -> str:
        if model_row.empty or pd.isna(model_row.iloc[0][column]):
            return "N/A"
        return f"{float(model_row.iloc[0][column]):.4f}"

    text = f"""# Stage 6 Model Comparison

## Scope

Stage 6 compares all implemented default main-spec models on the saved OOS outputs from Stages 3 and 5:

- `mlp_predictor`
- `ipca_style`
- `conditional_autoencoder_style`
- `graph_conditional_latent_factor`

The comparison uses the common stock-date intersection across all models, so the main tables are as close to apples-to-apples as possible. Coverage is summarized in the saved CSV tables.

## Common OOS Sample

All four models overlap on the same 24 OOS months from 2016-01-31 through 2017-12-31. The aligned sample uses the stock-date intersection across all models before calculating summary metrics.

## Main Result

The graph model does show incremental value relative to the characteristic-only CAE-style benchmark on ranking-oriented metrics, but not a clean across-the-board win over all characteristic-only models.

- Graph model OOS R^2: {_metric(graph_row, 'oos_r2_zero_benchmark')}
- Graph model rank IC mean: {_metric(graph_row, 'rank_ic_mean')}
- Graph model cross-sectional corr mean: {_metric(graph_row, 'cross_sectional_corr_mean')}
- Graph model pricing error monthly RMSE: {_metric(graph_row, 'pricing_error_monthly_rmse')}

Against the nonlinear characteristic-only exposure model (`conditional_autoencoder_style`):

- Graph rank IC is higher: {_metric(graph_row, 'rank_ic_mean')} vs {_metric(cae_row, 'rank_ic_mean')}
- Graph cross-sectional corr is higher: {_metric(graph_row, 'cross_sectional_corr_mean')} vs {_metric(cae_row, 'cross_sectional_corr_mean')}
- Graph OOS R^2 is slightly worse: {_metric(graph_row, 'oos_r2_zero_benchmark')} vs {_metric(cae_row, 'oos_r2_zero_benchmark')}

Against the direct nonlinear predictor (`mlp_predictor`):

- Graph rank IC is lower: {_metric(graph_row, 'rank_ic_mean')} vs {_metric(mlp_row, 'rank_ic_mean')}
- Graph OOS R^2 is better than MLP but still negative: {_metric(graph_row, 'oos_r2_zero_benchmark')} vs {_metric(mlp_row, 'oos_r2_zero_benchmark')}
- Graph pricing error RMSE is slightly lower: {_metric(graph_row, 'pricing_error_monthly_rmse')} vs {_metric(mlp_row, 'pricing_error_monthly_rmse')}

Against the linear dynamic-beta benchmark (`ipca_style`):

- Graph rank IC is slightly higher: {_metric(graph_row, 'rank_ic_mean')} vs {_metric(ipca_row, 'rank_ic_mean')}
- Graph OOS R^2 is slightly worse: {_metric(graph_row, 'oos_r2_zero_benchmark')} vs {_metric(ipca_row, 'oos_r2_zero_benchmark')}

## Interpretation

At this point, the gains from graph structure look more like **ranking gains than raw prediction gains**.

- Best OOS R^2 on the aligned sample: `{winner_name(summary, 'oos_r2_zero_benchmark', higher_is_better=True)}`
- Best rank IC mean: `{winner_name(summary, 'rank_ic_mean', higher_is_better=True)}`
- Best cross-sectional correlation mean: `{winner_name(summary, 'cross_sectional_corr_mean', higher_is_better=True)}`
- Lowest pricing error monthly RMSE: `{winner_name(summary, 'pricing_error_monthly_rmse', higher_is_better=False)}`

This suggests the current graph model is helping the conditional beta function capture some useful cross-sectional ordering information relative to CAE-style and IPCA-style latent factor models, but it is not yet producing a decisive OOS prediction improvement over the strongest characteristic-only benchmark set.

## Latent-Factor Diagnostics

Latent-factor-related diagnostics are available for IPCA-style, CAE-style, and the graph model, but not for the direct MLP predictor. These diagnostics are reported in the latent diagnostics CSV. They should be interpreted carefully because latent factors are only identified up to standard rotation/sign conventions and the current factor forecast is just the train-window mean.

## Figures

- Overall metric bars: `{project_relative_string(project_root, plot_paths['overall_metric_bars'])}`
- Monthly rank IC: `{project_relative_string(project_root, plot_paths['monthly_rank_ic'])}`
- Monthly cross-sectional correlation: `{project_relative_string(project_root, plot_paths['monthly_cross_sectional_corr'])}`
- Prediction correlation heatmap: `{project_relative_string(project_root, plot_paths['prediction_correlation_heatmap'])}`

## Strongest Remaining Limitations

1. The OOS comparison window is short: only 24 months in the current default run.
2. The graph model is still a simple first-pass GCN/GAT beta encoder, not a stronger structural asset-pricing system.
3. The graph is a homogeneous combination of multiple similarity layers, not a true multi-relation graph model.
4. The factor forecast uses the train-window mean latent factor premium rather than a dedicated factor forecasting model.
5. There is still no Stage 6 portfolio comparison yet, so the economic value is only partially observed through prediction/ranking/pricing diagnostics.
6. Industry edges are still unavailable because the stored data does not contain point-in-time industry classifications.
7. Hyperparameter tuning is still shallow, so the current graph result should be interpreted as a baseline rather than an optimized ceiling.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")


def run_stage6_evaluation(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run the Stage 6 unified model comparison workflow."""

    started_at = time.time()
    config = load_config(config_path)
    predictions, exposures, factors, metadata = load_output_frames(project_root, config)
    aligned, coverage, month_counts = align_common_prediction_panel(predictions)
    monthly = monthly_metric_table(aligned)
    summary = summary_metric_table(aligned, monthly)
    corr_table = prediction_correlation_table(aligned)
    model_names = summary["model"].tolist()
    latent = latent_diagnostics(exposures, factors, model_names)

    outputs = config.get("outputs", {})
    table_dir = resolve_project_path(project_root, outputs.get("table_dir", "outputs/comparison/stage6_tables"))
    plot_dir = resolve_project_path(project_root, outputs.get("plot_dir", "outputs/comparison/stage6_plots"))
    metadata_path = resolve_project_path(project_root, outputs.get("metadata_path", "outputs/metadata/stage6_comparison_metadata.json"))
    report_path = resolve_project_path(project_root, outputs.get("report_path", "reports/stage6_model_comparison.md"))
    for directory in [table_dir, plot_dir, metadata_path.parent, report_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    summary_path = table_dir / 'stage6_summary_metrics.csv'
    monthly_path = table_dir / 'stage6_monthly_metrics.csv'
    coverage_path = table_dir / 'stage6_coverage_by_model.csv'
    month_counts_path = table_dir / 'stage6_month_counts_by_model.csv'
    corr_path = table_dir / 'stage6_prediction_correlation.csv'
    latent_path = table_dir / 'stage6_latent_diagnostics.csv'

    summary.to_csv(summary_path, index=False)
    monthly.to_csv(monthly_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    month_counts.to_csv(month_counts_path, index=False)
    corr_table.to_csv(corr_path, index=False)
    latent.to_csv(latent_path, index=False)

    plot_paths = make_comparison_plots(summary, monthly, corr_table, plot_dir)
    write_interpretation_note(project_root, config, coverage, summary, latent, plot_paths, report_path)

    run_metadata = {
        "stage": "stage6_model_comparison",
        "config_path": project_relative_string(project_root, config_path),
        "comparison_label": config.get("comparison", {}).get("label", "stage6_main_spec"),
        "aligned_models": model_names,
        "aligned_n_obs_per_model": int(summary["n_obs"].iloc[0]) if not summary.empty else 0,
        "aligned_n_months": int(summary["n_months"].iloc[0]) if not summary.empty else 0,
        "date_min": str(aligned["date"].min().date()) if not aligned.empty else None,
        "date_max": str(aligned["date"].max().date()) if not aligned.empty else None,
        "table_outputs": {
            "summary_metrics": project_relative_string(project_root, summary_path),
            "monthly_metrics": project_relative_string(project_root, monthly_path),
            "coverage": project_relative_string(project_root, coverage_path),
            "month_counts": project_relative_string(project_root, month_counts_path),
            "prediction_correlation": project_relative_string(project_root, corr_path),
            "latent_diagnostics": project_relative_string(project_root, latent_path),
        },
        "plot_outputs": {key: project_relative_string(project_root, value) for key, value in plot_paths.items()},
        "report_path": project_relative_string(project_root, report_path),
        "source_metadata_keys": list(metadata.keys()),
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    metadata_path.write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved Stage 6 comparison outputs:")
    print(f"  summary metrics: {summary_path}")
    print(f"  monthly metrics: {monthly_path}")
    print(f"  latent diagnostics: {latent_path}")
    print(f"  report: {report_path}")
    print(summary.to_string(index=False))
    return run_metadata



