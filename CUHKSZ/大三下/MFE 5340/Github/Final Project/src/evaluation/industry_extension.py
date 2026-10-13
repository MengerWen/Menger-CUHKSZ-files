"""Industry-edge extension workflow for the final project.

This module keeps the original Stage 3-8 artifacts untouched. It builds
additional industry-only and industry-plus-similarity graph variants, trains
new graph latent-factor models, and writes extension-specific comparison,
portfolio, figure, and report artifacts.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.evaluation.model_comparison import align_common_prediction_panel, monthly_metric_table, summary_metric_table
from src.evaluation.stage8_analysis import portfolio_summary_for_predictions
from src.graphs.monthly_graphs import load_industry_labels, run_stage4_graph_construction
from src.training.graph_model_pipeline import run_stage5_graph_model
from src.training.non_graph_benchmark_pipeline import prediction_metrics, resolve_project_path


MODEL_LABELS = {
    "mlp_predictor": "MLP",
    "ipca_style": "IPCA-style",
    "conditional_autoencoder_style": "CAE-style",
    "graph_conditional_latent_factor": "Original graph",
    "graph_industry_only": "Industry-only graph",
    "graph_industry_hybrid": "Industry + hybrid graph",
}


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML config file."""

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {path}")
    return config


def save_yaml(config: dict[str, Any], path: Path) -> None:
    """Write one generated YAML config."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def project_relative_string(project_root: Path, path: Path | str) -> str:
    """Return a project-relative string when possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a mapping without mutating the input."""

    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value).date())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def audit_industry_labels(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit static industry-label coverage against the cleaned panel."""

    data_config = config.get("data", {})
    panel_path = resolve_project_path(project_root, data_config.get("panel_path", "outputs/panels/main_features500_panel.pkl"))
    industry_path = data_config.get("industry_label_path", "data/ind_code.pkl")
    panel = pd.read_pickle(panel_path).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    labels = load_industry_labels(project_root, industry_path)
    if labels is None or labels.empty:
        raise ValueError(f"no usable industry labels found at {industry_path}")

    panel_stocks = pd.Index(panel["stock_id"].astype(str).unique())
    covered_stock_count = int(panel_stocks.isin(labels.index).sum())
    working = panel[["date", "stock_id"]].copy()
    working["industry_code"] = working["stock_id"].astype(str).map(labels)

    rows: list[dict[str, Any]] = []
    for date, month in working.groupby("date", sort=True):
        counts = month.dropna(subset=["industry_code"]).groupby("industry_code")["stock_id"].nunique()
        clique_edges = int(((counts * (counts - 1)) // 2).sum()) if not counts.empty else 0
        rows.append(
            {
                "date": pd.Timestamp(date),
                "n_nodes": int(month["stock_id"].nunique()),
                "covered_nodes": int(month["industry_code"].notna().sum()),
                "coverage_ratio": float(month["industry_code"].notna().mean()) if len(month) else np.nan,
                "n_industries": int(counts.shape[0]),
                "industry_clique_edges": clique_edges,
                "max_industry_size": int(counts.max()) if not counts.empty else 0,
            }
        )
    monthly = pd.DataFrame(rows)
    summary = {
        "industry_label_path": project_relative_string(project_root, resolve_project_path(project_root, industry_path)),
        "label_name": str(labels.name),
        "label_rows": int(labels.shape[0]),
        "unique_industries": int(labels.nunique()),
        "panel_unique_stocks": int(panel_stocks.shape[0]),
        "covered_panel_stocks": covered_stock_count,
        "stock_coverage_ratio": float(covered_stock_count / panel_stocks.shape[0]) if panel_stocks.shape[0] else np.nan,
        "panel_rows": int(panel.shape[0]),
        "covered_panel_rows": int(working["industry_code"].notna().sum()),
        "row_coverage_ratio": float(working["industry_code"].notna().mean()) if len(working) else np.nan,
        "monthly_industry_count_min": int(monthly["n_industries"].min()),
        "monthly_industry_count_max": int(monthly["n_industries"].max()),
        "monthly_industry_clique_edges_mean": float(monthly["industry_clique_edges"].mean()),
        "monthly_industry_clique_edges_max": int(monthly["industry_clique_edges"].max()),
    }
    return monthly, summary


def generated_stage4_config(
    project_root: Path,
    base_config_path: Path,
    extension_root: Path,
    variant: dict[str, Any],
    industry_label_path: str,
) -> Path:
    """Create one extension-specific Stage 4 config."""

    name = variant["name"]
    base = load_yaml(base_config_path)
    graph_overrides = copy.deepcopy(variant.get("graph_overrides", {}))
    graph_overrides.setdefault("graph", {})
    graph_overrides["graph"]["industry_label_path"] = industry_label_path
    graph_overrides["graph"]["allow_missing_industry"] = False
    config = deep_update(base, graph_overrides)

    graph_dir = extension_root / "graphs" / variant.get("graph_dir_name", name)
    config.setdefault("outputs", {})
    config["outputs"]["graph_dir"] = project_relative_string(project_root, graph_dir)
    config["outputs"]["stats_path"] = project_relative_string(project_root, extension_root / "graphs" / f"{name}_stats.csv")
    config["outputs"]["manifest_path"] = project_relative_string(project_root, extension_root / "graphs" / f"{name}_manifest.csv")
    config["outputs"]["metadata_path"] = project_relative_string(project_root, extension_root / "metadata" / f"{name}_graph_metadata.json")

    path = extension_root / "generated_configs" / f"{name}_stage4.yaml"
    save_yaml(config, path)
    return path


def generated_stage5_config(
    project_root: Path,
    base_config_path: Path,
    extension_root: Path,
    variant: dict[str, Any],
    manifest_path: str,
) -> Path:
    """Create one extension-specific Stage 5 config."""

    name = variant["name"]
    base = load_yaml(base_config_path)
    config = deep_update(base, variant.get("model_overrides", {}))
    config.setdefault("graphs", {})
    config["graphs"]["manifest_path"] = manifest_path
    config.setdefault("outputs", {})
    config["outputs"]["predictions_path"] = project_relative_string(project_root, extension_root / "predictions" / f"{name}.pkl")
    config["outputs"]["exposures_path"] = project_relative_string(project_root, extension_root / "latent" / f"{name}_exposures.pkl")
    config["outputs"]["factors_path"] = project_relative_string(project_root, extension_root / "latent" / f"{name}_factors.pkl")
    config["outputs"]["attention_path"] = project_relative_string(project_root, extension_root / "attention" / f"{name}_attention.pkl")
    config["outputs"]["metrics_path"] = project_relative_string(project_root, extension_root / "metrics" / f"{name}.csv")
    config["outputs"]["metadata_path"] = project_relative_string(project_root, extension_root / "metadata" / f"{name}_model_metadata.json")

    path = extension_root / "generated_configs" / f"{name}_stage5.yaml"
    save_yaml(config, path)
    return path


def relabel_stage5_outputs(project_root: Path, model_name: str, model_metadata: dict[str, Any]) -> dict[str, str]:
    """Replace the generic Stage 5 model name in saved variant artifacts."""

    outputs = model_metadata["outputs"]
    for key in ["predictions", "exposures", "factors", "attention"]:
        path = resolve_project_path(project_root, outputs[key])
        frame = pd.read_pickle(path)
        if isinstance(frame, pd.DataFrame) and "model" in frame.columns:
            frame = frame.copy()
            frame["model"] = model_name
            frame.to_pickle(path)

    predictions_path = resolve_project_path(project_root, outputs["predictions"])
    metrics_path = resolve_project_path(project_root, outputs["metrics"])
    predictions = pd.read_pickle(predictions_path)
    metrics = prediction_metrics(predictions)
    metrics.to_csv(metrics_path, index=False)

    metadata_path = resolve_project_path(project_root, outputs["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["extension_model_name"] = model_name
    metadata["model_relabel_note"] = "Stage 5 generic graph model output was relabeled for the industry extension comparison."
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return {key: outputs[key] for key in ["predictions", "exposures", "factors", "attention", "metrics", "metadata"]}


def run_industry_variants(project_root: Path, config: dict[str, Any], extension_root: Path) -> list[dict[str, Any]]:
    """Build industry graph variants and train the corresponding graph models."""

    base = config.get("base_configs", {})
    base_stage4 = resolve_project_path(project_root, base.get("stage4_graph_config", "configs/graphs_features500.yaml"))
    base_stage5 = resolve_project_path(project_root, base.get("stage5_graph_model_config", "configs/graph_model_features500.yaml"))
    industry_path = config.get("data", {}).get("industry_label_path", "data/ind_code.pkl")
    results: list[dict[str, Any]] = []

    for variant in config.get("industry_variants", []):
        name = variant["name"]
        print(f"[industry extension] building graph variant: {name}")
        stage4_config = generated_stage4_config(project_root, base_stage4, extension_root, variant, industry_path)
        graph_meta = run_stage4_graph_construction(stage4_config, project_root)
        print(f"[industry extension] training model variant: {name}")
        stage5_config = generated_stage5_config(project_root, base_stage5, extension_root, variant, graph_meta["outputs"]["manifest_path"])
        model_meta = run_stage5_graph_model(stage5_config, project_root)
        relabeled_outputs = relabel_stage5_outputs(project_root, name, model_meta)
        results.append(
            {
                "name": name,
                "label": variant.get("label", name),
                "stage4_config": project_relative_string(project_root, stage4_config),
                "stage5_config": project_relative_string(project_root, stage5_config),
                "graph_metadata": graph_meta,
                "model_metadata": model_meta,
                "outputs": relabeled_outputs,
            }
        )
    return results


def load_predictions_for_specs(project_root: Path, model_specs: list[dict[str, str]]) -> pd.DataFrame:
    """Load and model-label prediction outputs."""

    frames: list[pd.DataFrame] = []
    for spec in model_specs:
        frame = pd.read_pickle(resolve_project_path(project_root, spec["predictions_path"])).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if "model" in frame.columns and spec["name"] in set(frame["model"].astype(str)):
            frame = frame.loc[frame["model"].astype(str) == spec["name"]].copy()
        if frame.empty:
            raise ValueError(f"no prediction rows found for model {spec['name']} in {spec['predictions_path']}")
        frame["model"] = spec["name"]
        frames.append(frame[["date", "stock_id", "model", "y_true", "y_pred"]])
    return pd.concat(frames, ignore_index=True)


def normalized_prediction_specs(project_root: Path, model_specs: list[dict[str, str]], output_dir: Path) -> list[dict[str, str]]:
    """Write one model-specific prediction file per spec for downstream helpers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[dict[str, str]] = []
    for spec in model_specs:
        frame = pd.read_pickle(resolve_project_path(project_root, spec["predictions_path"])).copy()
        frame["date"] = pd.to_datetime(frame["date"])
        if "model" in frame.columns and spec["name"] in set(frame["model"].astype(str)):
            frame = frame.loc[frame["model"].astype(str) == spec["name"]].copy()
        if frame.empty:
            raise ValueError(f"no prediction rows found for model {spec['name']} in {spec['predictions_path']}")
        frame["model"] = spec["name"]
        path = output_dir / f"{spec['name']}.pkl"
        frame.to_pickle(path)
        normalized.append({"name": spec["name"], "predictions_path": project_relative_string(project_root, path)})
    return normalized


def run_prediction_comparison(
    project_root: Path,
    model_specs: list[dict[str, str]],
    table_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Compute Stage 6-style prediction and pricing diagnostics."""

    predictions = load_predictions_for_specs(project_root, model_specs)
    aligned, coverage, month_counts = align_common_prediction_panel(predictions)
    monthly = monthly_metric_table(aligned)
    summary = summary_metric_table(aligned, monthly)
    table_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_dir / "industry_model_comparison.csv", index=False)
    monthly.to_csv(table_dir / "industry_monthly_metrics.csv", index=False)
    coverage.to_csv(table_dir / "industry_coverage_by_model.csv", index=False)
    month_counts.to_csv(table_dir / "industry_month_counts_by_model.csv", index=False)
    return {"predictions": predictions, "aligned": aligned, "coverage": coverage, "monthly": monthly, "summary": summary}


def _pick_portfolio_metrics(performance: pd.DataFrame, main_cost_bps: int) -> pd.DataFrame:
    """Create a compact portfolio table used in the report."""

    main = performance.loc[performance["transaction_cost_bps"] == main_cost_bps].copy()
    picks = [
        ("long_only", "value", "annualized_return", "long_only_value_ann_return"),
        ("long_only", "value", "sharpe_ratio", "long_only_value_sharpe"),
        ("long_short", "equal", "annualized_return", "long_short_equal_ann_return"),
        ("long_short", "equal", "sharpe_ratio", "long_short_equal_sharpe"),
        ("long_short", "equal", "max_drawdown", "long_short_equal_max_drawdown"),
        ("long_only", "value", "avg_monthly_turnover", "long_only_value_avg_turnover"),
    ]
    rows: list[dict[str, Any]] = []
    for model, group in main.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model}
        for strategy, weight, column, out_col in picks:
            block = group.loc[(group["strategy_name"] == strategy) & (group["weight_scheme"] == weight)]
            row[out_col] = float(block.iloc[0][column]) if not block.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def plot_prediction_extension(summary: pd.DataFrame, figure_dir: Path, dpi: int) -> dict[str, str]:
    """Create the report-facing industry prediction comparison figure."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [model for model in MODEL_LABELS if model in set(summary["model"])]
    frame = summary.copy()
    frame["model"] = pd.Categorical(frame["model"], categories=order, ordered=True)
    frame = frame.sort_values("model")
    labels = [MODEL_LABELS.get(str(model), str(model)) for model in frame["model"]]
    metrics = [
        ("oos_r2_zero_benchmark", "OOS R2"),
        ("rank_ic_mean", "Rank IC"),
        ("cross_sectional_corr_mean", "CS Corr"),
        ("pricing_error_monthly_rmse", "Pricing RMSE"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, metrics, strict=True):
        ax.bar(labels, frame[column])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("")
    base = figure_dir / "figure_9_industry_graph_extension"
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"figure_9_industry_graph_extension": str(base.with_suffix(".png"))}


def plot_portfolio_extension(compact: pd.DataFrame, figure_dir: Path, dpi: int) -> dict[str, str]:
    """Create the report-facing industry portfolio comparison figure."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    order = [model for model in MODEL_LABELS if model in set(compact["model"])]
    frame = compact.copy()
    frame["model"] = pd.Categorical(frame["model"], categories=order, ordered=True)
    frame = frame.sort_values("model")
    labels = [MODEL_LABELS.get(str(model), str(model)) for model in frame["model"]]
    metrics = [
        ("long_only_value_ann_return", "Long-only VW annualized return"),
        ("long_only_value_sharpe", "Long-only VW Sharpe"),
        ("long_short_equal_ann_return", "Long-short EW annualized return"),
        ("long_short_equal_sharpe", "Long-short EW Sharpe"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, metrics, strict=True):
        ax.bar(labels, frame[column])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("")
    base = figure_dir / "figure_10_industry_portfolio_extension"
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {"figure_10_industry_portfolio_extension": str(base.with_suffix(".png"))}


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Render a compact Markdown table without optional dependencies."""

    rows = frame[columns].copy()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows.itertuples(index=False):
        lines.append("| " + " | ".join(_format_value(value) for value in row) + " |")
    return "\n".join(lines)


def write_extension_report(
    project_root: Path,
    report_path: Path,
    audit_summary: dict[str, Any],
    prediction_summary: pd.DataFrame,
    portfolio_compact: pd.DataFrame,
    figure_paths: dict[str, str],
    main_cost_bps: int,
) -> None:
    """Write the industry extension Markdown report."""

    graph_rows = prediction_summary.loc[
        prediction_summary["model"].isin(["graph_conditional_latent_factor", "graph_industry_only", "graph_industry_hybrid"])
    ].copy()
    portfolio_rows = portfolio_compact.loc[
        portfolio_compact["model"].isin(["graph_conditional_latent_factor", "graph_industry_only", "graph_industry_hybrid"])
    ].copy()

    lines = [
        "# Industry Classification Extension",
        "",
        "## Scope",
        "",
        "This extension adds the teacher-provided static first-level industry classification `data/ind_code.pkl` as an additional graph-relation source. It does not overwrite the original Stage 3-8 outputs. Industry codes are used only to form graph edges, not as extra model features.",
        "",
        "Because the file has no date dimension, the industry relation is treated as a static industry prior rather than a point-in-time changing industry history.",
        "",
        "## Industry Data Audit",
        "",
        f"- Label file: `{audit_summary['industry_label_path']}`",
        f"- Label name: `{audit_summary['label_name']}`",
        f"- Covered panel stocks: {audit_summary['covered_panel_stocks']} / {audit_summary['panel_unique_stocks']} ({audit_summary['stock_coverage_ratio']:.2%})",
        f"- Covered panel rows: {audit_summary['covered_panel_rows']} / {audit_summary['panel_rows']} ({audit_summary['row_coverage_ratio']:.2%})",
        f"- Monthly industry count range: {audit_summary['monthly_industry_count_min']} to {audit_summary['monthly_industry_count_max']}",
        f"- Average same-industry clique edges per month: {audit_summary['monthly_industry_clique_edges_mean']:.2f}",
        "",
        "## Added Model Variants",
        "",
        "- `graph_industry_only`: uses only same-industry edges.",
        "- `graph_industry_hybrid`: uses the original dynamic similarity layers plus same-industry edges.",
        "",
        "## Prediction and Pricing Comparison",
        "",
        markdown_table(
            graph_rows,
            ["model", "oos_r2_zero_benchmark", "rank_ic_mean", "cross_sectional_corr_mean", "pricing_error_monthly_rmse"],
        ),
        "",
        "## Portfolio Comparison",
        "",
        f"The portfolio table uses the main transaction-cost setting of `{main_cost_bps}` bps.",
        "",
        markdown_table(
            portfolio_rows,
            ["model", "long_only_value_ann_return", "long_only_value_sharpe", "long_short_equal_ann_return", "long_short_equal_sharpe"],
        ),
        "",
        "## Figures",
        "",
    ]
    for name, path in figure_paths.items():
        lines.append(f"- `{name}`: `{project_relative_string(project_root, path)}`")
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "If `graph_industry_hybrid` improves on the original graph model, the static industry relation is adding useful structure beyond dynamic similarity. If `graph_industry_only` is weaker but `graph_industry_hybrid` is competitive, industry is best read as a stable prior rather than a complete substitute for dynamic graph information. If neither industry variant improves the result, the original similarity-hybrid graph remains valuable because it is not merely recreating broad industry clusters.",
            "",
            "## Limitations",
            "",
            "1. The industry classification is static, so it cannot capture historical industry reclassifications.",
            "2. The extension keeps the same short 24-month OOS window as the original main comparison.",
            "3. The graph encoder is still the same compact GCN latent-factor model; this extension changes relation structure, not the core pricing architecture.",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_industry_extension(config_path: Path, project_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Run the full additive industry extension workflow."""

    started_at = time.time()
    config = load_yaml(config_path)
    outputs = config.get("outputs", {})
    extension_root = resolve_project_path(project_root, outputs.get("root", "outputs/industry_extension"))
    if smoke:
        extension_root = extension_root.with_name(extension_root.name + "_smoke")
        for variant in config.get("industry_variants", []):
            variant.setdefault("graph_overrides", {}).setdefault("graph", {})["max_months"] = int(config.get("smoke", {}).get("graph_max_months", 120))
            variant.setdefault("model_overrides", {}).setdefault("oos", {})["max_oos_months"] = int(config.get("smoke", {}).get("max_oos_months", 2))

    table_dir = extension_root / "tables"
    plot_dir = extension_root / "plots"
    metadata_dir = extension_root / "metadata"
    figure_dir = resolve_project_path(project_root, outputs.get("figure_dir", "reports/figures"))
    report_path = resolve_project_path(project_root, outputs.get("report_path", "reports/industry_extension_results.md"))
    if smoke:
        figure_dir = extension_root / "figures"
        report_path = extension_root / "industry_extension_results_smoke.md"
    for directory in [extension_root, table_dir, plot_dir, metadata_dir, figure_dir, report_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    audit_monthly, audit_summary = audit_industry_labels(project_root, config)
    audit_monthly_path = table_dir / "industry_coverage_by_month.csv"
    audit_monthly.to_csv(audit_monthly_path, index=False)

    variant_results = run_industry_variants(project_root, config, extension_root)
    existing_specs = config.get("existing_models", [])
    new_specs = [{"name": result["name"], "predictions_path": result["outputs"]["predictions"]} for result in variant_results]
    all_specs = [*existing_specs, *new_specs]
    normalized_specs = normalized_prediction_specs(project_root, all_specs, extension_root / "normalized_predictions")

    prediction_outputs = run_prediction_comparison(project_root, normalized_specs, table_dir)
    base_stage7 = resolve_project_path(project_root, config.get("base_configs", {}).get("stage7_portfolio_config", "configs/portfolio_features500.yaml"))
    portfolio_outputs = portfolio_summary_for_predictions(project_root, base_stage7, normalized_specs, plot_dir=plot_dir / "portfolio")
    portfolio_summary_path = table_dir / "industry_portfolio_summary.csv"
    portfolio_outputs["performance"].to_csv(portfolio_summary_path, index=False)
    portfolio_outputs["monthly_returns"].to_pickle(extension_root / "industry_portfolio_monthly_returns.pkl")
    portfolio_outputs["weights"].to_pickle(extension_root / "industry_portfolio_weights.pkl")

    main_cost = int(config.get("portfolio", {}).get("main_transaction_cost_bps", 10))
    portfolio_compact = _pick_portfolio_metrics(portfolio_outputs["performance"], main_cost)
    portfolio_compact_path = table_dir / "industry_portfolio_compact.csv"
    portfolio_compact.to_csv(portfolio_compact_path, index=False)

    dpi = int(config.get("figures", {}).get("dpi", 200))
    figure_paths = {}
    figure_paths.update(plot_prediction_extension(prediction_outputs["summary"], figure_dir, dpi))
    figure_paths.update(plot_portfolio_extension(portfolio_compact, figure_dir, dpi))
    write_extension_report(project_root, report_path, audit_summary, prediction_outputs["summary"], portfolio_compact, figure_paths, main_cost)

    metadata = {
        "stage": "industry_extension",
        "config_path": project_relative_string(project_root, config_path),
        "smoke": smoke,
        "industry_audit": audit_summary,
        "variant_results": variant_results,
        "model_specs": normalized_specs,
        "outputs": {
            "root": project_relative_string(project_root, extension_root),
            "industry_coverage_by_month": project_relative_string(project_root, audit_monthly_path),
            "industry_model_comparison": project_relative_string(project_root, table_dir / "industry_model_comparison.csv"),
            "industry_portfolio_summary": project_relative_string(project_root, portfolio_summary_path),
            "industry_portfolio_compact": project_relative_string(project_root, portfolio_compact_path),
            "report": project_relative_string(project_root, report_path),
            "figures": {key: project_relative_string(project_root, value) for key, value in figure_paths.items()},
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
    }
    metadata_path = metadata_dir / "industry_extension_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print("Saved industry extension outputs:")
    print(f"  comparison: {table_dir / 'industry_model_comparison.csv'}")
    print(f"  portfolio:  {portfolio_summary_path}")
    print(f"  report:     {report_path}")
    print(prediction_outputs["summary"].to_string(index=False))
    return metadata
