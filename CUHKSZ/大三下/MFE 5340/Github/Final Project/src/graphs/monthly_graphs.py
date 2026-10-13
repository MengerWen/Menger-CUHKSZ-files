"""Monthly stock graph construction for Stage 4."""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.graphs.build_graph import edges_to_pyg_data

INDUSTRY_SEARCH_TERMS = ("industry", "sector", "class", "classification", "citic", "sw", "ind_code", "申万", "中信")


@dataclass(frozen=True)
class MonthlyGraphConfig:
    """Core graph construction choices for one Stage 4 run."""

    edge_types: tuple[str, ...]
    return_lookback_months: int
    min_return_observations: int
    include_current_month_return: bool
    k_return: int
    k_feature_cosine: int
    k_feature_euclidean: int
    min_edge_weight: float
    combine_rule: str
    save_pyg: bool
    max_months: int | None
    industry_label_path: str | None
    industry_edge_weight: float
    industry_min_group_size: int


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config."""

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"config did not load to a mapping: {config_path}")
    return config


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Resolve project-relative paths."""

    path = Path(value)
    return path if path.is_absolute() else project_root / path


def project_relative_string(project_root: Path, path: Path | str) -> str:
    """Return a stable project-relative path string when possible."""

    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(candidate)


def parse_graph_config(config: dict[str, Any]) -> MonthlyGraphConfig:
    """Normalize graph configuration values."""

    graph = config.get("graph", {})
    edge_types = tuple(graph.get("edge_types", ["return_correlation", "feature_cosine_knn", "feature_euclidean_knn"]))
    allowed = {"return_correlation", "feature_cosine_knn", "feature_euclidean_knn", "industry"}
    unknown = sorted(set(edge_types) - allowed)
    if unknown:
        raise ValueError(f"unknown graph edge types: {unknown}")
    industry_label_path = graph.get("industry_label_path") or config.get("data", {}).get("industry_label_path")
    if "industry" in edge_types and not industry_label_path and not graph.get("allow_missing_industry", False):
        raise ValueError("industry edges requested; set graph.industry_label_path or data.industry_label_path")

    return MonthlyGraphConfig(
        edge_types=edge_types,
        return_lookback_months=int(graph.get("return_lookback_months", 12)),
        min_return_observations=int(graph.get("min_return_observations", 6)),
        include_current_month_return=bool(graph.get("include_current_month_return", True)),
        k_return=int(graph.get("k_return", 10)),
        k_feature_cosine=int(graph.get("k_feature_cosine", 10)),
        k_feature_euclidean=int(graph.get("k_feature_euclidean", 10)),
        min_edge_weight=float(graph.get("min_edge_weight", 0.0)),
        combine_rule=str(graph.get("combine_rule", "mean")),
        save_pyg=bool(graph.get("save_pyg", True)),
        max_months=graph.get("max_months"),
        industry_label_path=str(industry_label_path) if industry_label_path else None,
        industry_edge_weight=float(graph.get("industry_edge_weight", 1.0)),
        industry_min_group_size=int(graph.get("industry_min_group_size", 2)),
    )


def _clean_industry_code_values(values: pd.Series) -> pd.Series:
    """Normalize raw industry labels to stable string codes."""

    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        return numeric.astype("Int64").astype(str)
    return values.astype(str).str.strip()


def load_industry_labels(project_root: Path, industry_label_path: str | Path | None) -> pd.Series | None:
    """Load static stock-level industry labels indexed by stock id."""

    if not industry_label_path:
        return None
    path = resolve_project_path(project_root, industry_label_path)
    raw = pd.read_pickle(path)
    if isinstance(raw, pd.Series):
        labels = raw.copy()
        labels.index = labels.index.astype(str)
        labels.index.name = "stock_id"
        labels = labels.dropna()
        labels = _clean_industry_code_values(labels)
    elif isinstance(raw, pd.DataFrame):
        frame = raw.copy()
        stock_candidates = [col for col in ["stock_id", "order_book_id", "asset", "ticker"] if col in frame.columns]
        industry_candidates = [col for col in ["industry_code", "first_industry_code", "industry", "sector"] if col in frame.columns]
        if not stock_candidates or not industry_candidates:
            raise ValueError(f"could not infer stock and industry columns from {path}")
        labels = frame.set_index(stock_candidates[0])[industry_candidates[0]].dropna()
        labels.index = labels.index.astype(str)
        labels.index.name = "stock_id"
        labels = _clean_industry_code_values(labels)
    else:
        raise TypeError(f"unsupported industry label object type in {path}: {type(raw).__name__}")
    labels.name = "industry_code"
    return labels


def inspect_industry_availability(project_root: Path, panel: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check whether explicit industry/sector labels are present in stored data."""

    config = config or {}
    configured_path = config.get("graph", {}).get("industry_label_path") or config.get("data", {}).get("industry_label_path")
    data_dir = project_root / "data"
    matching_data_files = [
        path.relative_to(project_root).as_posix()
        for path in data_dir.rglob("*")
        if path.is_file() and any(term in path.name.lower() for term in INDUSTRY_SEARCH_TERMS)
    ]
    matching_panel_columns = [col for col in panel.columns if any(term in str(col).lower() for term in INDUSTRY_SEARCH_TERMS)]
    matching_feature_files: list[str] = []
    for feature_dir in [data_dir / "features500", data_dir / "features"]:
        if feature_dir.exists():
            matching_feature_files.extend(
                path.relative_to(project_root).as_posix()
                for path in feature_dir.glob("*.pkl")
                if any(term in path.name.lower() for term in INDUSTRY_SEARCH_TERMS)
            )
    hdf_price_columns: list[str] = []
    price_path = data_dir / "price.h5"
    if price_path.exists():
        try:
            with pd.HDFStore(price_path, mode="r") as store:
                if "/price" in store.keys():
                    columns = store.get("/price").columns
                    hdf_price_columns = [col for col in columns if any(term in str(col).lower() for term in INDUSTRY_SEARCH_TERMS)]
        except Exception as exc:  # pragma: no cover - defensive inspection branch
            hdf_price_columns = [f"price.h5 inspection failed: {type(exc).__name__}: {exc}"]

    configured_audit: dict[str, Any] = {}
    if configured_path:
        labels = load_industry_labels(project_root, configured_path)
        if labels is not None:
            panel_stocks = pd.Index(panel["stock_id"].astype(str).unique())
            covered_stocks = int(panel_stocks.isin(labels.index).sum())
            row_coverage = panel["stock_id"].astype(str).isin(labels.index)
            configured_audit = {
                "configured_industry_label_path": project_relative_string(project_root, resolve_project_path(project_root, configured_path)),
                "label_object": "Series or DataFrame normalized to stock_id -> industry_code",
                "label_rows": int(labels.shape[0]),
                "unique_industries": int(labels.nunique()),
                "covered_panel_stocks": covered_stocks,
                "panel_unique_stocks": int(panel_stocks.shape[0]),
                "stock_coverage_ratio": float(covered_stocks / panel_stocks.shape[0]) if panel_stocks.shape[0] else np.nan,
                "covered_panel_rows": int(row_coverage.sum()),
                "panel_rows": int(panel.shape[0]),
                "row_coverage_ratio": float(row_coverage.mean()) if len(row_coverage) else np.nan,
            }

    available = bool(configured_audit or matching_data_files or matching_panel_columns or matching_feature_files or hdf_price_columns)
    return {
        "explicit_industry_labels_available": available,
        "matching_data_files": matching_data_files,
        "matching_panel_columns": matching_panel_columns,
        "matching_feature_files": matching_feature_files,
        "matching_price_columns": hdf_price_columns,
        "configured_industry_audit": configured_audit,
        "conclusion": "No explicit industry/sector labels found in stored Stage 4 inputs." if not available else "Potential industry fields found; inspect before use.",
    }


def load_stage4_inputs(project_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[str]]:
    """Load cleaned panel, monthly returns, metadata, and feature columns."""

    data_config = config.get("data", {})
    panel_path = resolve_project_path(project_root, data_config.get("panel_path", "outputs/panels/main_features500_panel.pkl"))
    metadata_path = resolve_project_path(project_root, data_config.get("metadata_path", "outputs/metadata/main_features500_panel_metadata.json"))
    returns_path = resolve_project_path(project_root, data_config.get("monthly_returns_path", "data/monthly_returns.pkl"))

    panel = pd.read_pickle(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    feature_cols = metadata.get("features", {}).get("kept_features")
    if not feature_cols:
        raise ValueError("could not infer feature columns from Stage 2 metadata")
    missing = [col for col in ["date", "stock_id", "mcap_t", *feature_cols] if col not in panel.columns]
    if missing:
        raise KeyError(f"panel is missing required graph columns: {missing[:20]}")

    returns = pd.read_pickle(returns_path)
    returns.index = pd.to_datetime(returns.index)
    returns.columns = returns.columns.astype(str)
    return panel, returns, metadata, list(feature_cols)


def apply_node_filters(month_frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Optionally filter graph nodes using known month-t market cap or liquidity."""

    filters = config.get("node_filters", {})
    frame = month_frame.copy()
    mcap_filter = filters.get("market_cap", {})
    if mcap_filter.get("enabled", False):
        quantile = float(mcap_filter.get("min_quantile", 0.0))
        threshold = frame["mcap_t"].quantile(quantile)
        frame = frame.loc[frame["mcap_t"] >= threshold].copy()

    liquidity_filter = filters.get("liquidity", {})
    if liquidity_filter.get("enabled", False):
        liquidity_col = str(liquidity_filter.get("column", "amount_21"))
        if liquidity_col not in frame.columns:
            raise KeyError(f"liquidity filter column is unavailable in panel: {liquidity_col}")
        quantile = float(liquidity_filter.get("min_quantile", 0.0))
        threshold = frame[liquidity_col].quantile(quantile)
        frame = frame.loc[frame[liquidity_col] >= threshold].copy()
    return frame


def _dedupe_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate undirected edges within one edge type."""

    if edges.empty:
        return edges
    ordered = np.sort(edges[["source", "target"]].astype(str).to_numpy(), axis=1)
    out = edges.copy()
    out["source"] = ordered[:, 0]
    out["target"] = ordered[:, 1]
    out = out.loc[out["source"] != out["target"]]
    if out.empty:
        return out
    return (
        out.groupby(["source", "target", "edge_type"], as_index=False)
        .agg(weight=("weight", "max"), distance=("distance", "min"))
        .sort_values(["edge_type", "source", "target"])
        .reset_index(drop=True)
    )


def return_correlation_edges(
    returns: pd.DataFrame,
    graph_month: pd.Timestamp,
    stocks: list[str],
    graph_config: MonthlyGraphConfig,
) -> pd.DataFrame:
    """Build positive return-correlation kNN edges using returns available through month t."""

    end_month = graph_month if graph_config.include_current_month_return else graph_month - pd.offsets.MonthEnd(1)
    start_month = end_month - pd.offsets.MonthEnd(graph_config.return_lookback_months - 1)
    window = returns.loc[(returns.index >= start_month) & (returns.index <= end_month), returns.columns.intersection(stocks)].copy()
    if window.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])
    valid_counts = window.notna().sum(axis=0)
    valid_stocks = valid_counts[valid_counts >= graph_config.min_return_observations].index.astype(str).tolist()
    if len(valid_stocks) <= 1:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])

    clean = window[valid_stocks].apply(pd.to_numeric, errors="coerce")
    corr = clean.corr(min_periods=graph_config.min_return_observations).fillna(0.0).clip(-1.0, 1.0)
    corr = corr.where(corr > graph_config.min_edge_weight, other=0.0)
    np.fill_diagonal(corr.values, 0.0)

    records: list[dict[str, Any]] = []
    k = min(graph_config.k_return, max(0, corr.shape[0] - 1))
    if k <= 0:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])
    for source in corr.index.astype(str):
        neighbors = corr.loc[source].sort_values(ascending=False).head(k)
        for target, weight in neighbors.items():
            weight = float(weight)
            if target == source or weight <= graph_config.min_edge_weight:
                continue
            records.append({"source": source, "target": str(target), "edge_type": "return_correlation", "weight": weight, "distance": 1.0 - weight})
    return _dedupe_edges(pd.DataFrame.from_records(records, columns=["source", "target", "edge_type", "weight", "distance"]))


def feature_knn_edges(
    month_frame: pd.DataFrame,
    feature_cols: list[str],
    edge_type: str,
    k_neighbors: int,
    metric: str,
    min_edge_weight: float,
) -> pd.DataFrame:
    """Build feature-based kNN edges from month-t characteristics."""

    stocks = month_frame["stock_id"].astype(str).to_numpy()
    x = month_frame[feature_cols].to_numpy(dtype=np.float32, copy=True)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if len(stocks) <= 1:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])

    if metric == "cosine":
        x_model = normalize(x, norm="l2", axis=1)
    elif metric == "euclidean":
        x_model = x
    else:
        raise ValueError(f"unsupported feature kNN metric: {metric}")

    k = min(k_neighbors + 1, len(stocks))
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=1)
    nn.fit(x_model)
    distances, indices = nn.kneighbors(x_model)

    records: list[dict[str, Any]] = []
    for src_pos, source in enumerate(stocks):
        for dst_pos, distance in zip(indices[src_pos][1:], distances[src_pos][1:]):
            target = stocks[int(dst_pos)]
            distance = float(distance)
            if metric == "cosine":
                weight = 1.0 - distance
            else:
                weight = float(1.0 / (1.0 + distance))
            if weight <= min_edge_weight or source == target:
                continue
            records.append({"source": str(source), "target": str(target), "edge_type": edge_type, "weight": weight, "distance": distance})
    return _dedupe_edges(pd.DataFrame.from_records(records, columns=["source", "target", "edge_type", "weight", "distance"]))


def industry_edges(
    stocks: list[str],
    industry_labels: pd.Series | None,
    edge_weight: float,
    min_group_size: int,
) -> pd.DataFrame:
    """Build static same-industry clique edges for the current stock universe."""

    if industry_labels is None:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])
    label_frame = pd.DataFrame({"stock_id": stocks})
    label_frame["industry_code"] = label_frame["stock_id"].map(industry_labels)
    label_frame = label_frame.dropna(subset=["industry_code"])
    if label_frame.empty:
        return pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])

    records: list[dict[str, Any]] = []
    for _industry_code, group in label_frame.groupby("industry_code", sort=True):
        members = sorted(group["stock_id"].astype(str).tolist())
        if len(members) < min_group_size:
            continue
        for source, target in itertools.combinations(members, 2):
            records.append(
                {
                    "source": source,
                    "target": target,
                    "edge_type": "industry",
                    "weight": float(edge_weight),
                    "distance": float(1.0 - edge_weight),
                }
            )
    return _dedupe_edges(pd.DataFrame.from_records(records, columns=["source", "target", "edge_type", "weight", "distance"]))


def combine_edges(edges: pd.DataFrame, combine_rule: str) -> pd.DataFrame:
    """Combine multiple edge types into one homogeneous graph for first-pass GNN use."""

    if edges.empty:
        return pd.DataFrame(columns=["source", "target", "weight", "distance", "edge_types", "edge_type_count"])
    if combine_rule not in {"mean", "max"}:
        raise ValueError("combine_rule must be 'mean' or 'max'")
    weight_agg = "mean" if combine_rule == "mean" else "max"
    combined = (
        edges.groupby(["source", "target"], as_index=False)
        .agg(
            weight=("weight", weight_agg),
            distance=("distance", "min"),
            edge_types=("edge_type", lambda values: "+".join(sorted(set(map(str, values))))),
            edge_type_count=("edge_type", lambda values: len(set(map(str, values)))),
        )
        .sort_values(["source", "target"])
        .reset_index(drop=True)
    )
    return combined


def graph_stats(date: pd.Timestamp, node_count: int, typed_edges: pd.DataFrame, combined_edges: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive graph stats for one month."""

    rows: list[dict[str, Any]] = []
    edge_layers = [("combined", combined_edges)]
    if not typed_edges.empty:
        edge_layers.extend(list(typed_edges.groupby("edge_type", sort=True)))
    for label, edges in edge_layers:
        edge_count = int(edges.shape[0])
        rows.append(
            {
                "date": date,
                "edge_layer": label,
                "n_nodes": node_count,
                "n_edges": edge_count,
                "avg_degree_undirected": float(2.0 * edge_count / node_count) if node_count else np.nan,
                "density": float(2.0 * edge_count / (node_count * (node_count - 1))) if node_count > 1 else np.nan,
                "mean_weight": float(edges["weight"].mean()) if edge_count and "weight" in edges else np.nan,
                "median_weight": float(edges["weight"].median()) if edge_count and "weight" in edges else np.nan,
                "min_weight": float(edges["weight"].min()) if edge_count and "weight" in edges else np.nan,
                "max_weight": float(edges["weight"].max()) if edge_count and "weight" in edges else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _month_filename(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def build_graph_for_month(
    month_frame: pd.DataFrame,
    returns: pd.DataFrame,
    feature_cols: list[str],
    graph_config: MonthlyGraphConfig,
    config: dict[str, Any],
    industry_labels: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build typed edges, combined edges, node features, and stats for one month."""

    graph_month = pd.Timestamp(month_frame["date"].iloc[0])
    filtered = apply_node_filters(month_frame, config).sort_values("stock_id").reset_index(drop=True)
    stocks = filtered["stock_id"].astype(str).tolist()
    edge_frames: list[pd.DataFrame] = []

    if "return_correlation" in graph_config.edge_types:
        edge_frames.append(return_correlation_edges(returns, graph_month, stocks, graph_config))
    if "feature_cosine_knn" in graph_config.edge_types:
        edge_frames.append(feature_knn_edges(filtered, feature_cols, "feature_cosine_knn", graph_config.k_feature_cosine, "cosine", graph_config.min_edge_weight))
    if "feature_euclidean_knn" in graph_config.edge_types:
        edge_frames.append(feature_knn_edges(filtered, feature_cols, "feature_euclidean_knn", graph_config.k_feature_euclidean, "euclidean", graph_config.min_edge_weight))
    if "industry" in graph_config.edge_types:
        edge_frames.append(industry_edges(stocks, industry_labels, graph_config.industry_edge_weight, graph_config.industry_min_group_size))

    typed_edges = pd.concat(edge_frames, ignore_index=True) if edge_frames else pd.DataFrame(columns=["source", "target", "edge_type", "weight", "distance"])
    combined_edges = combine_edges(typed_edges, graph_config.combine_rule)
    node_features = filtered.set_index("stock_id")[feature_cols].astype("float32")
    stats = graph_stats(graph_month, node_features.shape[0], typed_edges, combined_edges)
    return typed_edges, combined_edges, node_features, stats


def run_stage4_graph_construction(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run the Stage 4 monthly graph construction pipeline."""

    started_at = time.time()
    config = load_yaml_config(config_path)
    graph_config = parse_graph_config(config)
    panel, returns, panel_metadata, feature_cols = load_stage4_inputs(project_root, config)
    industry_labels = load_industry_labels(project_root, graph_config.industry_label_path)
    industry_audit = inspect_industry_availability(project_root, panel, config)
    if industry_audit["explicit_industry_labels_available"]:
        print("Potential industry labels found; inspect audit before enabling industry edges.")
    else:
        print("No explicit industry labels found. Stage 4 will build similarity-based graphs only.")
    if "industry" in graph_config.edge_types and industry_labels is None:
        raise ValueError("industry edges requested, but no industry labels were loaded")

    output_config = config.get("outputs", {})
    graph_dir = resolve_project_path(project_root, output_config.get("graph_dir", "outputs/graphs/features500_similarity"))
    edges_dir = graph_dir / "edges"
    pyg_dir = graph_dir / "pyg"
    stats_path = resolve_project_path(project_root, output_config.get("stats_path", "outputs/graphs/features500_similarity_graph_stats.csv"))
    manifest_path = resolve_project_path(project_root, output_config.get("manifest_path", "outputs/graphs/features500_similarity_manifest.csv"))
    metadata_path = resolve_project_path(project_root, output_config.get("metadata_path", "outputs/metadata/stage4_graph_metadata.json"))
    for directory in [edges_dir, pyg_dir, stats_path.parent, manifest_path.parent, metadata_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    months = pd.DatetimeIndex(panel["date"].drop_duplicates()).sort_values()
    if graph_config.max_months is not None:
        months = months[: int(graph_config.max_months)]

    stats_frames: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for month in months:
        month_frame = panel.loc[panel["date"] == month].copy()
        typed_edges, combined_edges, node_features, stats = build_graph_for_month(month_frame, returns, feature_cols, graph_config, config, industry_labels=industry_labels)
        month_name = _month_filename(month)
        edges_path = edges_dir / f"{month_name}_edges.pkl"
        pyg_path = pyg_dir / f"{month_name}.pt"
        edge_payload = {
            "date": month,
            "typed_edges": typed_edges,
            "combined_edges": combined_edges,
            "node_features": node_features,
            "graph_config": asdict(graph_config),
        }
        pd.to_pickle(edge_payload, edges_path)
        if graph_config.save_pyg:
            pyg_data = edges_to_pyg_data(combined_edges, node_features)
            pyg_data.date = month_name
            pyg_data.edge_types = list(graph_config.edge_types)
            pyg_data.combine_rule = graph_config.combine_rule
            torch.save(pyg_data, pyg_path)
        else:
            pyg_path = Path("")
        stats_frames.append(stats)
        manifest_rows.append(
            {
                "date": month,
                "n_nodes": int(node_features.shape[0]),
                "n_typed_edges": int(typed_edges.shape[0]),
                "n_combined_edges": int(combined_edges.shape[0]),
                "edge_path": project_relative_string(project_root, edges_path),
                "pyg_path": project_relative_string(project_root, pyg_path) if graph_config.save_pyg else "",
            }
        )
        print(f"built {month_name}: nodes={node_features.shape[0]} typed_edges={typed_edges.shape[0]} combined_edges={combined_edges.shape[0]}")

    stats_frame = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    manifest = pd.DataFrame(manifest_rows)
    stats_frame.to_csv(stats_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    feature_universe = panel_metadata.get("config", {}).get("feature_universe") or panel_metadata.get("feature_universe")
    run_metadata: dict[str, Any] = {
        "stage": "stage4_monthly_graphs",
        "config_path": project_relative_string(project_root, config_path),
        "feature_universe": feature_universe,
        "feature_count": len(feature_cols),
        "panel_rows": int(panel.shape[0]),
        "panel_months_available": int(panel["date"].nunique()),
        "months_built": int(len(months)),
        "date_start": str(months.min().date()) if len(months) else None,
        "date_end": str(months.max().date()) if len(months) else None,
        "graph_config": asdict(graph_config),
        "industry_audit": industry_audit,
        "outputs": {
            "graph_dir": project_relative_string(project_root, graph_dir),
            "stats_path": project_relative_string(project_root, stats_path),
            "manifest_path": project_relative_string(project_root, manifest_path),
            "metadata_path": project_relative_string(project_root, metadata_path),
        },
        "elapsed_seconds": round(time.time() - started_at, 3),
        "lookahead_control": "For graph month t, feature edges use month-t cleaned characteristics and return-correlation edges use monthly returns through t when include_current_month_return is true. No t+1 target returns are used.",
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2, ensure_ascii=False)

    print("Saved Stage 4 graph outputs:")
    print(f"  graph_dir: {graph_dir}")
    print(f"  stats:     {stats_path}")
    print(f"  manifest:  {manifest_path}")
    print(f"  metadata:  {metadata_path}")
    if not stats_frame.empty:
        summary = stats_frame.groupby("edge_layer")["n_edges"].describe().round(3)
        print(summary.to_string())
    return run_metadata





