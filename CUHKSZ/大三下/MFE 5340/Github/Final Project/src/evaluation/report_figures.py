"""Generate report-facing figures from saved project outputs."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

sns.set_theme(style="whitegrid", context="talk")

MODEL_ORDER = [
    "mlp_predictor",
    "ipca_style",
    "conditional_autoencoder_style",
    "graph_conditional_latent_factor",
]
MODEL_LABELS = {
    "mlp_predictor": "MLP",
    "ipca_style": "IPCA-style",
    "conditional_autoencoder_style": "CAE-style",
    "graph_conditional_latent_factor": "Graph model",
}
MODEL_COLORS = {
    "mlp_predictor": "#4C78A8",
    "ipca_style": "#F58518",
    "conditional_autoencoder_style": "#E45756",
    "graph_conditional_latent_factor": "#54A24B",
}
GRAPH_VARIANT_ORDER = [
    "graph_main_diagnostic",
    "graph_return_only",
    "graph_lookback6",
    "graph_latent_k5",
    "graph_gat_hybrid",
    "graph_conditional_latent_factor_static_test_graph",
]
GRAPH_VARIANT_LABELS = {
    "graph_main_diagnostic": "Main hybrid GCN",
    "graph_return_only": "Return-only graph",
    "graph_lookback6": "Hybrid graph, lookback 6",
    "graph_latent_k5": "Hybrid graph, K=5",
    "graph_gat_hybrid": "Hybrid GAT",
    "graph_conditional_latent_factor_static_test_graph": "Static test graph",
}
GRAPH_VARIANT_COLORS = {
    "graph_main_diagnostic": "#54A24B",
    "graph_return_only": "#4C78A8",
    "graph_lookback6": "#F58518",
    "graph_latent_k5": "#E45756",
    "graph_gat_hybrid": "#72B7B2",
    "graph_conditional_latent_factor_static_test_graph": "#9D9D9D",
}
EDGE_TYPE_LABELS = {
    "feature_cosine_knn": "Feature cosine kNN",
    "feature_euclidean_knn": "Feature euclidean kNN",
    "return_correlation": "Return correlation",
    "combined": "Combined",
}
MAIN_FIGURE_IDS = [
    "figure_1_sample_coverage",
    "figure_2_graph_overview",
    "figure_3_model_comparison",
    "figure_4_portfolio_cumulative",
    "figure_5_portfolio_summary",
    "figure_7_graph_robustness",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping config at {path}")
    return data


def resolve_project_path(project_root: Path, project_relative_path: str) -> Path:
    path = Path(project_relative_path)
    if path.is_absolute():
        return path
    return project_root / path


def project_relative_string(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def wrap_label(value: str, width: int = 26) -> str:
    return textwrap.fill(value.replace("_", " "), width=width)


def format_date_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=45)


def metric_title(metric: str) -> str:
    titles = {
        "oos_r2_zero_benchmark": "OOS R^2",
        "rank_ic_mean": "Rank IC",
        "cross_sectional_corr_mean": "Cross-sectional Correlation",
        "pricing_error_monthly_rmse": "Pricing Error RMSE",
    }
    return titles.get(metric, metric)


def value_format(metric: str) -> str:
    return "{:.3f}" if "rmse" in metric or "corr" in metric or "rank" in metric or "r2" in metric else "{:.2f}"


def inspection_summary(project_root: Path) -> dict[str, Any]:
    buckets = {
        "stage6_plots": project_root / "outputs/comparison/stage6_plots",
        "stage7_plots": project_root / "outputs/portfolio/stage7_plots",
        "stage8_plots": project_root / "outputs/stage8/plots",
        "reports": project_root / "reports",
    }
    results: dict[str, Any] = {}
    for name, folder in buckets.items():
        pngs = sorted(project_relative_string(project_root, path) for path in folder.rglob("*.png")) if folder.exists() else []
        results[name] = {
            "path": project_relative_string(project_root, folder),
            "png_count": len(pngs),
            "examples": pngs[:6],
        }
    results["missing_before_this_stage"] = [
        "Git-tracked report figure package under reports/figures/",
        "Data and sample overview figure for stock coverage and raw feature missingness.",
        "Graph construction overview figure for nodes, combined edges, and edge-layer averages.",
        "A unified figure guide mapping charts to source files and report interpretation.",
    ]
    return results


def load_inputs(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    panel = pd.read_pickle(resolve_project_path(project_root, paths["panel_path"])).copy()
    panel["date"] = pd.to_datetime(panel["date"])

    metadata_path = resolve_project_path(project_root, paths["panel_metadata_path"])
    with metadata_path.open("r", encoding="utf-8") as file:
        panel_metadata = json.load(file)

    graph_manifest = pd.read_csv(resolve_project_path(project_root, paths["graph_manifest_path"]))
    graph_manifest["date"] = pd.to_datetime(graph_manifest["date"])

    graph_stats = pd.read_csv(resolve_project_path(project_root, paths["graph_stats_path"]))
    graph_stats["date"] = pd.to_datetime(graph_stats["date"])

    stage6_summary = pd.read_csv(resolve_project_path(project_root, paths["stage6_summary_path"]))
    stage6_monthly = pd.read_csv(resolve_project_path(project_root, paths["stage6_monthly_path"]))
    stage6_monthly["date"] = pd.to_datetime(stage6_monthly["date"])

    stage7_performance = pd.read_csv(resolve_project_path(project_root, paths["stage7_performance_path"]))
    stage7_returns = pd.read_pickle(resolve_project_path(project_root, paths["stage7_monthly_returns_path"]))
    if not isinstance(stage7_returns, pd.DataFrame):
        raise TypeError("Expected stage7_monthly_returns.pkl to be a pandas DataFrame")
    for column in ["signal_date", "holding_month"]:
        if column in stage7_returns.columns:
            stage7_returns[column] = pd.to_datetime(stage7_returns[column])

    stage8_main = pd.read_csv(resolve_project_path(project_root, paths["stage8_main_results_path"]))
    feature_links = pd.read_csv(resolve_project_path(project_root, paths["stage8_feature_links_path"]))
    permutation = pd.read_csv(resolve_project_path(project_root, paths["stage8_permutation_path"]))
    neighbor_edge_mix = pd.read_csv(resolve_project_path(project_root, paths["stage8_neighbor_edge_mix_path"]))
    if "date" in neighbor_edge_mix.columns:
        neighbor_edge_mix["date"] = pd.to_datetime(neighbor_edge_mix["date"])
    robustness_summary = pd.read_csv(resolve_project_path(project_root, paths["stage8_robustness_summary_path"]))
    robustness_portfolio = pd.read_csv(resolve_project_path(project_root, paths["stage8_robustness_portfolio_path"]))
    gat_attention = read_optional_csv(resolve_project_path(project_root, paths["stage8_gat_attention_path"]))
    if not gat_attention.empty and "date" in gat_attention.columns:
        gat_attention["date"] = pd.to_datetime(gat_attention["date"])

    return {
        "panel": panel,
        "panel_metadata": panel_metadata,
        "graph_manifest": graph_manifest,
        "graph_stats": graph_stats,
        "stage6_summary": stage6_summary,
        "stage6_monthly": stage6_monthly,
        "stage7_performance": stage7_performance,
        "stage7_returns": stage7_returns,
        "stage8_main": stage8_main,
        "feature_links": feature_links,
        "permutation": permutation,
        "neighbor_edge_mix": neighbor_edge_mix,
        "robustness_summary": robustness_summary,
        "robustness_portfolio": robustness_portfolio,
        "gat_attention": gat_attention,
    }

def figure_record(
    project_root: Path,
    *,
    figure_id: str,
    title: str,
    classification: str,
    sources: list[str],
    interpretation: str,
    output_base: Path,
) -> dict[str, Any]:
    return {
        "id": figure_id,
        "title": title,
        "classification": classification,
        "sources": sources,
        "interpretation": interpretation,
        "png": project_relative_string(project_root, output_base.with_suffix(".png")),
        "pdf": project_relative_string(project_root, output_base.with_suffix(".pdf")),
    }


def save_figure(fig: plt.Figure, output_base: Path, dpi: int) -> None:
    ensure_parent(output_base.with_suffix(".png"))
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def ordered_main_results(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.loc[frame["model"].isin(MODEL_ORDER)].copy()
    ordered["model"] = pd.Categorical(ordered["model"], categories=MODEL_ORDER, ordered=True)
    ordered = ordered.sort_values("model").reset_index(drop=True)
    ordered["model_label"] = ordered["model"].map(MODEL_LABELS)
    ordered["color"] = ordered["model"].map(MODEL_COLORS)
    return ordered


def annotate_bars(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    ymin, ymax = ax.get_ylim()
    scale = ymax - ymin if ymax != ymin else 1.0
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height):
            continue
        y = height + 0.03 * scale if height >= 0 else height - 0.06 * scale
        va = "bottom" if height >= 0 else "top"
        ax.text(patch.get_x() + patch.get_width() / 2.0, y, fmt.format(height), ha="center", va=va, fontsize=9)


def plot_sample_coverage(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    panel = data["panel"]
    metadata = data["panel_metadata"]

    counts = panel.groupby("date", sort=True)["stock_id"].nunique().reset_index(name="n_stocks")
    missing = metadata.get("features", {}).get("raw_missing_top10", {})
    missing_df = pd.DataFrame({"feature": list(missing.keys()), "missing_rate": list(missing.values())})
    missing_df = missing_df.sort_values("missing_rate", ascending=True).tail(int(settings["top_missing_features"]))
    missing_df["feature_label"] = missing_df["feature"].map(lambda x: wrap_label(str(x), 24))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(counts["date"], counts["n_stocks"], color="#4C78A8", linewidth=2.4)
    ax.set_title("Monthly Stock Coverage")
    ax.set_ylabel("Number of stocks")
    ax.set_xlabel("Month")
    format_date_axis(ax)
    ax.axhline(counts["n_stocks"].median(), color="#9D9D9D", linestyle="--", linewidth=1.2)
    ax.text(
        counts["date"].iloc[-1],
        counts["n_stocks"].median() + 5,
        f"Median = {counts['n_stocks'].median():.0f}",
        ha="right",
        va="bottom",
        fontsize=10,
        color="#555555",
    )

    ax = axes[1]
    if missing_df.empty:
        ax.text(0.5, 0.5, "No raw missingness summary found", ha="center", va="center")
        ax.axis("off")
    else:
        ax.barh(missing_df["feature_label"], 100.0 * missing_df["missing_rate"], color="#E45756")
        ax.set_title("Top Raw Feature Missingness")
        ax.set_xlabel("Missing rate (%)")
        ax.set_ylabel("")

    output_base = figure_dir / "figure_1_sample_coverage"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_1_sample_coverage",
        title="Sample Coverage and Raw Feature Missingness",
        classification="main",
        sources=[config["paths"]["panel_path"], config["paths"]["panel_metadata_path"]],
        interpretation="The left panel shows the active monthly stock universe in the main-spec panel. The right panel shows which saved raw characteristics were most sparse before cleaning, which helps frame the feature engineering burden behind the benchmark and graph models.",
        output_base=output_base,
    )


def plot_graph_overview(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    manifest = data["graph_manifest"].copy()
    stats = data["graph_stats"].copy()

    layer_means = (
        stats.loc[stats["edge_layer"].isin(["return_correlation", "feature_cosine_knn", "feature_euclidean_knn"])]
        .groupby("edge_layer", as_index=False)["n_edges"]
        .mean()
    )
    layer_means["edge_layer_label"] = layer_means["edge_layer"].map(EDGE_TYPE_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(manifest["date"], manifest["n_nodes"], label="Nodes", color="#4C78A8", linewidth=2.2)
    ax.set_title("Monthly Graph Size")
    ax.set_ylabel("Nodes")
    ax.set_xlabel("Month")
    format_date_axis(ax)

    ax2 = ax.twinx()
    ax2.plot(manifest["date"], manifest["n_combined_edges"], label="Combined edges", color="#F58518", linewidth=2.2)
    ax2.set_ylabel("Combined undirected edges")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left", frameon=True)

    ax = axes[1]
    ax.bar(layer_means["edge_layer_label"], layer_means["n_edges"], color=["#72B7B2", "#54A24B", "#F58518"])
    ax.set_title("Average Monthly Edge Count by Layer")
    ax.set_ylabel("Mean undirected edges")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)

    output_base = figure_dir / "figure_2_graph_overview"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_2_graph_overview",
        title="Graph Construction Overview",
        classification="main",
        sources=[config["paths"]["graph_manifest_path"], config["paths"]["graph_stats_path"]],
        interpretation="The left panel shows how the monthly graph size evolves over time. The right panel compares the average sparsified edge count contributed by each implemented edge layer, which makes the hybrid graph design concrete.",
        output_base=output_base,
    )


def plot_model_comparison(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    results = ordered_main_results(data["stage8_main"])
    metrics = [
        "oos_r2_zero_benchmark",
        "rank_ic_mean",
        "cross_sectional_corr_mean",
        "pricing_error_monthly_rmse",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics, strict=True):
        ax.bar(results["model_label"], results[metric], color=results["color"].tolist())
        ax.set_title(metric_title(metric))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        annotate_bars(ax, fmt=value_format(metric))
        ax.set_ylabel("Lower is better" if metric == "pricing_error_monthly_rmse" else "Higher is better")

    output_base = figure_dir / "figure_3_model_comparison"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_3_model_comparison",
        title="Main Benchmark Comparison",
        classification="main",
        sources=[config["paths"]["stage8_main_results_path"]],
        interpretation="This figure is the apples-to-apples benchmark comparison on the aligned main-spec OOS sample. It shows that the graph model improves some cross-sectional metrics relative to characteristic-only latent models, but not every metric simultaneously.",
        output_base=output_base,
    )

def build_cumulative_series(frame: pd.DataFrame) -> pd.DataFrame:
    series = frame.sort_values("holding_month").copy()
    series["wealth"] = (1.0 + series["net_return"].fillna(0.0)).cumprod()
    series["cumulative_return"] = series["wealth"] - 1.0
    return series


def plot_portfolio_cumulative(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    monthly = data["stage7_returns"].copy()
    cost = int(settings["main_transaction_cost_bps"])
    lo_weight = settings["long_only_weight_scheme"]
    ls_weight = settings["long_short_weight_scheme"]

    long_only = monthly.loc[
        (monthly["transaction_cost_bps"] == cost)
        & (monthly["strategy_name"] == "long_only")
        & (monthly["weight_scheme"] == lo_weight)
        & (monthly["model"].isin(MODEL_ORDER))
    ].copy()
    long_short = monthly.loc[
        (monthly["transaction_cost_bps"] == cost)
        & (monthly["strategy_name"] == "long_short")
        & (monthly["weight_scheme"] == ls_weight)
        & (monthly["model"].isin(MODEL_ORDER))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    for ax, frame, title in [
        (axes[0], long_only, f"Long-only ({str(lo_weight).upper()}), {cost} bps"),
        (axes[1], long_short, f"Long-short ({str(ls_weight).upper()}), {cost} bps"),
    ]:
        for model in MODEL_ORDER:
            model_frame = frame.loc[frame["model"] == model].copy()
            if model_frame.empty:
                continue
            model_frame = build_cumulative_series(model_frame)
            ax.plot(
                model_frame["holding_month"],
                model_frame["cumulative_return"],
                label=MODEL_LABELS[model],
                color=MODEL_COLORS[model],
                linewidth=2.2,
            )
        ax.set_title(title)
        ax.set_ylabel("Cumulative return")
        ax.set_xlabel("Holding month")
        format_date_axis(ax)
        ax.axhline(0.0, color="#999999", linewidth=1.0)

    axes[0].legend(loc="upper left", frameon=True)
    output_base = figure_dir / "figure_4_portfolio_cumulative"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_4_portfolio_cumulative",
        title="Cumulative Portfolio Performance",
        classification="main",
        sources=[config["paths"]["stage7_monthly_returns_path"]],
        interpretation="These are the investable cumulative return paths on the default main transaction-cost setting. The left panel matches the main long-only result used in the project summary, while the right panel shows whether the graph signal also survives in a market-neutral sorting test.",
        output_base=output_base,
    )


def plot_portfolio_summary(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    performance = data["stage7_performance"].copy()
    cost = int(settings["main_transaction_cost_bps"])
    lo_weight = settings["long_only_weight_scheme"]
    ls_weight = settings["long_short_weight_scheme"]
    selections = [
        ("long_only", lo_weight, "Long-only annualized return", "annualized_return"),
        ("long_short", ls_weight, "Long-short annualized return", "annualized_return"),
        ("long_only", lo_weight, "Long-only Sharpe ratio", "sharpe_ratio"),
        ("long_short", ls_weight, "Long-short Sharpe ratio", "sharpe_ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, (strategy, weight, title, metric) in zip(axes.flat, selections, strict=True):
        frame = performance.loc[
            (performance["transaction_cost_bps"] == cost)
            & (performance["strategy_name"] == strategy)
            & (performance["weight_scheme"] == weight)
            & (performance["model"].isin(MODEL_ORDER))
        ].copy()
        frame = ordered_main_results(frame)
        ax.bar(frame["model_label"], frame[metric], color=frame["color"].tolist())
        ax.set_title(f"{title} ({str(weight).upper()}, {cost} bps)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        annotate_bars(ax, fmt="{:.3f}")

    output_base = figure_dir / "figure_5_portfolio_summary"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_5_portfolio_summary",
        title="Portfolio Return and Sharpe Summary",
        classification="main",
        sources=[config["paths"]["stage7_performance_path"]],
        interpretation="This summary figure turns the portfolio backtest into clean cross-model comparisons at the main cost setting. It helps separate whether the graph model wins through higher return, better risk control, or both.",
        output_base=output_base,
    )


def plot_interpretability(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    links = data["feature_links"].copy()
    permutation = data["permutation"].copy()
    edge_mix = data["neighbor_edge_mix"].copy()

    top_n = int(settings["top_heatmap_rows_per_model"])
    selected_links = (
        links.loc[links["model"].isin(["ipca_style", "conditional_autoencoder_style", "graph_conditional_latent_factor"])]
        .sort_values(["model", "max_abs_spearman"], ascending=[True, False])
        .groupby("model", group_keys=False)
        .head(top_n)
        .copy()
    )
    heatmap_df = (
        selected_links.pivot_table(index="feature", columns="model", values="max_abs_spearman", aggfunc="max")
        .reindex(columns=["ipca_style", "conditional_autoencoder_style", "graph_conditional_latent_factor"])
        .fillna(0.0)
    )
    heatmap_df = heatmap_df.loc[heatmap_df.max(axis=1).sort_values(ascending=False).index]
    heatmap_df.columns = [MODEL_LABELS.get(column, column) for column in heatmap_df.columns]
    heatmap_df.index = [wrap_label(str(index), 26) for index in heatmap_df.index]

    perm_top = permutation.sort_values("rank_ic_drop", ascending=False).head(int(settings["top_permutation_features"])).copy()
    perm_top = perm_top.sort_values("rank_ic_drop", ascending=True)
    perm_top["feature_label"] = perm_top["feature"].map(lambda x: wrap_label(str(x), 26))

    edge_share = edge_mix.groupby("edge_type", as_index=False)["share"].mean().sort_values("share", ascending=False)
    edge_share["edge_type_label"] = edge_share["edge_type"].map(lambda x: EDGE_TYPE_LABELS.get(str(x), str(x)))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)
    ax = axes[0]
    sns.heatmap(
        heatmap_df,
        cmap="YlGnBu",
        linewidths=0.5,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Abs monthly Spearman"},
        ax=ax,
    )
    ax.set_title("Top Feature-to-Exposure Links")
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax = axes[1]
    ax.barh(perm_top["feature_label"], perm_top["rank_ic_drop"], color="#54A24B")
    ax.set_title("Graph Permutation Importance")
    ax.set_xlabel("Rank IC drop")
    ax.set_ylabel("")

    ax = axes[2]
    palette = ["#72B7B2", "#54A24B", "#F58518"][: len(edge_share)]
    ax.bar(edge_share["edge_type_label"], edge_share["share"], color=palette)
    ax.set_title("Average Edge Mix Around Top Picks")
    ax.set_ylabel("Mean share")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)

    output_base = figure_dir / "figure_6_interpretability"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_6_interpretability",
        title="Interpretability and Graph Neighborhood Diagnostics",
        classification="main",
        sources=[
            config["paths"]["stage8_feature_links_path"],
            config["paths"]["stage8_permutation_path"],
            config["paths"]["stage8_neighbor_edge_mix_path"],
        ],
        interpretation="The heatmap shows which saved features align most strongly with latent exposures across the implemented conditional models. The permutation chart and edge-mix chart then focus on what seems to matter most for the main graph model and what kind of neighborhood structure its top-ranked names sit in.",
        output_base=output_base,
    )

def plot_graph_robustness(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any]:
    settings = config["settings"]
    summary = data["robustness_summary"].copy()
    portfolio = data["robustness_portfolio"].copy()

    summary = summary.loc[summary["model"].isin(GRAPH_VARIANT_ORDER)].copy()
    summary["model"] = pd.Categorical(summary["model"], categories=GRAPH_VARIANT_ORDER, ordered=True)
    summary = summary.sort_values("model")
    summary["label"] = summary["model"].map(GRAPH_VARIANT_LABELS)
    summary["color"] = summary["model"].map(GRAPH_VARIANT_COLORS)

    portfolio = portfolio.loc[portfolio["model"].isin(GRAPH_VARIANT_ORDER)].copy()
    portfolio["model"] = pd.Categorical(portfolio["model"], categories=GRAPH_VARIANT_ORDER, ordered=True)
    portfolio = portfolio.sort_values("model")
    portfolio["label"] = portfolio["model"].map(GRAPH_VARIANT_LABELS)
    portfolio["color"] = portfolio["model"].map(GRAPH_VARIANT_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), constrained_layout=True)
    ax = axes[0]
    x = np.arange(len(summary))
    width = 0.38
    ax.bar(x - width / 2.0, summary["rank_ic_mean"], width=width, color=summary["color"].tolist(), alpha=0.95, label="Rank IC")
    ax.bar(x + width / 2.0, summary["cross_sectional_corr_mean"], width=width, color=summary["color"].tolist(), alpha=0.55, label="CS corr")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(label, 16) for label in summary["label"]], rotation=15)
    ax.set_title("Prediction Robustness Across Graph Variants")
    ax.set_ylabel("Metric value")
    ax.legend(frameon=True)

    ax = axes[1]
    x = np.arange(len(portfolio))
    ax.bar(x - width / 2.0, portfolio["long_only_value_sharpe"], width=width, color=portfolio["color"].tolist(), alpha=0.95, label="Long-only VW Sharpe")
    ax.bar(x + width / 2.0, portfolio["long_short_equal_sharpe"], width=width, color=portfolio["color"].tolist(), alpha=0.55, label="Long-short EW Sharpe")
    ax.set_xticks(x)
    ax.set_xticklabels([wrap_label(label, 16) for label in portfolio["label"]], rotation=15)
    ax.set_title("Portfolio Robustness Across Graph Variants")
    ax.set_ylabel("Sharpe ratio")
    ax.legend(frameon=True)

    output_base = figure_dir / "figure_7_graph_robustness"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_7_graph_robustness",
        title="Graph Robustness Checks",
        classification="main",
        sources=[
            config["paths"]["stage8_robustness_summary_path"],
            config["paths"]["stage8_robustness_portfolio_path"],
        ],
        interpretation="This figure keeps the robustness section focused on a small number of strong checks. It shows whether the main hybrid graph remains the best choice once we vary the graph definition, latent dimension, or test-time graph behavior.",
        output_base=output_base,
    )


def plot_attention_exploratory(project_root: Path, data: dict[str, Any], config: dict[str, Any], figure_dir: Path) -> dict[str, Any] | None:
    settings = config["settings"]
    attention = data["gat_attention"].copy()
    if attention.empty or "edge_types" not in attention.columns or "attention_weight" not in attention.columns:
        return None

    summary = (
        attention.groupby("edge_types", as_index=False)
        .agg(mean_attention_weight=("attention_weight", "mean"), n_edges=("attention_weight", "size"))
        .sort_values(["n_edges", "mean_attention_weight"], ascending=[False, False])
        .head(8)
        .copy()
    )
    summary["edge_types_label"] = summary["edge_types"].map(lambda x: wrap_label(str(x).replace("+", " + "), 20))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    ax = axes[0]
    ax.bar(summary["edge_types_label"], summary["mean_attention_weight"], color="#72B7B2")
    ax.set_title("Mean GAT Attention by Edge Type")
    ax.set_ylabel("Mean attention weight")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    ax.bar(summary["edge_types_label"], summary["n_edges"], color="#4C78A8")
    ax.set_title("Most Frequent Attention Edge Types")
    ax.set_ylabel("Edge count")
    ax.tick_params(axis="x", rotation=20)

    output_base = figure_dir / "figure_8_gat_attention_exploratory"
    save_figure(fig, output_base, dpi=int(settings["dpi"]))
    return figure_record(
        project_root,
        figure_id="figure_8_gat_attention_exploratory",
        title="Exploratory GAT Attention Summary",
        classification="exploratory",
        sources=[config["paths"]["stage8_gat_attention_path"]],
        interpretation="This figure is exploratory because the main Stage 5 result is still a GCN run. It shows which saved edge-type combinations receive the largest average attention weights in the successful GAT robustness run.",
        output_base=output_base,
    )


def write_figure_guide(project_root: Path, guide_path: Path, inspection: dict[str, Any], figures: list[dict[str, Any]]) -> None:
    ensure_parent(guide_path)
    lines = ["# Figure Guide", "", "## Existing Plot Inventory", ""]
    for name, info in inspection.items():
        if name == "missing_before_this_stage":
            continue
        lines.append(f"- `{name}`: {info['png_count']} PNG files found under `{info['path']}`.")
        if info["examples"]:
            for example in info["examples"][:3]:
                lines.append(f"  - Example: `{example}`")
    lines.extend(["", "## Important Gaps Before This Stage", ""])
    for item in inspection.get("missing_before_this_stage", []):
        lines.append(f"- {item}")

    lines.extend(["", "## Created Report Figures", ""])
    for figure in figures:
        lines.append(f"### {figure['title']}")
        lines.append("")
        lines.append(f"- Figure id: `{figure['id']}`")
        lines.append(f"- Classification: `{figure['classification']}`")
        lines.append(f"- PNG: `{figure['png']}`")
        lines.append(f"- PDF: `{figure['pdf']}`")
        lines.append("- Source files:")
        for source in figure["sources"]:
            lines.append(f"  - `{source}`")
        lines.append(f"- How to read it: {figure['interpretation']}")
        lines.append("")

    lines.extend([
        "## Main vs Exploratory",
        "",
        "Main result figures are the figures marked `main`. They are the charts that best support the project narrative in the final write-up.",
        "",
        "Exploratory figures are supportive diagnostics that are useful in an appendix or discussion section, but should not carry the main claim by themselves.",
        "",
        "## Recommended Final Report Set",
        "",
    ])
    for figure_id in MAIN_FIGURE_IDS:
        match = next((figure for figure in figures if figure["id"] == figure_id), None)
        if match is not None:
            lines.append(f"- `{match['id']}`: {match['title']}")

    guide_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_report_figure_pipeline(config_path: Path, project_root: Path) -> dict[str, Any]:
    config = load_yaml(config_path)
    figure_dir = resolve_project_path(project_root, config["output"]["figure_dir"])
    guide_path = resolve_project_path(project_root, config["output"]["guide_path"])
    metadata_path = resolve_project_path(project_root, config["output"]["metadata_path"])
    figure_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(guide_path)
    ensure_parent(metadata_path)

    inspection = inspection_summary(project_root)
    data = load_inputs(project_root, config)

    figures = [
        plot_sample_coverage(project_root, data, config, figure_dir),
        plot_graph_overview(project_root, data, config, figure_dir),
        plot_model_comparison(project_root, data, config, figure_dir),
        plot_portfolio_cumulative(project_root, data, config, figure_dir),
        plot_portfolio_summary(project_root, data, config, figure_dir),
        plot_interpretability(project_root, data, config, figure_dir),
        plot_graph_robustness(project_root, data, config, figure_dir),
    ]
    attention_figure = plot_attention_exploratory(project_root, data, config, figure_dir)
    if attention_figure is not None:
        figures.append(attention_figure)

    write_figure_guide(project_root, guide_path, inspection, figures)

    metadata = {
        "config_path": project_relative_string(project_root, config_path),
        "inspection": inspection,
        "figures": figures,
        "recommended_main_figures": MAIN_FIGURE_IDS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Existing plot inventory:")
    for name, info in inspection.items():
        if name == "missing_before_this_stage":
            continue
        print(f"  {name}: {info['png_count']} png files in {info['path']}")
    print("Missing before this stage:")
    for item in inspection.get("missing_before_this_stage", []):
        print(f"  - {item}")
    print("Saved report-facing figures:")
    for figure in figures:
        print(f"  - {figure['png']}")
    print(f"Saved figure guide: {project_relative_string(project_root, guide_path)}")
    print(f"Saved figure manifest: {project_relative_string(project_root, metadata_path)}")
    return metadata
