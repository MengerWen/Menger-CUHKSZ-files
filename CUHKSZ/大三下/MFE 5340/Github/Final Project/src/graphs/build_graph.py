"""Graph construction helpers using pandas, scikit-learn, networkx, and PyG."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data


@dataclass(frozen=True)
class GraphSpec:
    """Configuration for a stock graph snapshot."""

    method: str = "return_correlation_knn"
    lookback_months: int = 12
    k_neighbors: int = 10
    include_industry_edges: bool = False


def validate_graph_spec(spec: GraphSpec) -> None:
    """Validate graph construction choices before using historical data."""

    if spec.method != "return_correlation_knn":
        raise ValueError("Only return_correlation_knn is implemented at this stage")
    if spec.lookback_months <= 0:
        raise ValueError("lookback_months must be positive")
    if spec.k_neighbors <= 0:
        raise ValueError("k_neighbors must be positive")


def correlation_knn_edges(returns_window: pd.DataFrame, spec: GraphSpec) -> pd.DataFrame:
    """Build undirected kNN edges from a stock return correlation matrix."""

    validate_graph_spec(spec)
    if returns_window.empty:
        raise ValueError("returns_window must be non-empty")

    clean = returns_window.apply(pd.to_numeric, errors="coerce")
    clean = clean.fillna(clean.median()).fillna(0.0)
    corr = clean.corr().fillna(0.0).clip(-1.0, 1.0)
    distance = 1.0 - corr
    np.fill_diagonal(distance.values, 0.0)

    n_neighbors = min(spec.k_neighbors + 1, distance.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed")
    nn.fit(distance)
    distances, indices = nn.kneighbors(distance)

    stocks = pd.Index(distance.index.astype(str), name="stock_id")
    records: list[dict[str, object]] = []
    for src_pos, src in enumerate(stocks):
        for dst_pos, dist in zip(indices[src_pos][1:], distances[src_pos][1:]):
            dst = stocks[dst_pos]
            if src == dst:
                continue
            a, b = sorted([src, dst])
            records.append({"source": a, "target": b, "distance": float(dist), "weight": float(1.0 - dist)})
    return pd.DataFrame.from_records(records).drop_duplicates(["source", "target"]).reset_index(drop=True)


def edges_to_networkx(edges: pd.DataFrame) -> nx.Graph:
    """Convert an edge DataFrame to a networkx Graph."""

    graph = nx.Graph()
    for row in edges.itertuples(index=False):
        graph.add_edge(row.source, row.target, weight=row.weight, distance=row.distance)
    return graph


def edges_to_pyg_data(edges: pd.DataFrame, node_features: pd.DataFrame) -> Data:
    """Convert edge and node feature DataFrames into a torch_geometric Data object.

    `node_features` must be indexed by stock id and contain numeric columns.
    """

    if not {"source", "target"}.issubset(edges.columns):
        raise KeyError("edges must contain source and target columns")
    if node_features.empty:
        raise ValueError("node_features must be non-empty")

    features = node_features.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    stock_ids = pd.Index(features.index.astype(str), name="stock_id")
    node_lookup = {stock: pos for pos, stock in enumerate(stock_ids)}
    usable_edges = edges.loc[edges["source"].isin(node_lookup) & edges["target"].isin(node_lookup)].copy()
    if usable_edges.empty:
        raise ValueError("no edges connect stocks present in node_features")

    edge_pairs = np.asarray(
        [
            [node_lookup[source], node_lookup[target]]
            for source, target in usable_edges[["source", "target"]].itertuples(index=False, name=None)
        ],
        dtype=np.int64,
    ).T
    reverse_pairs = edge_pairs[::-1]
    edge_index = torch.as_tensor(np.concatenate([edge_pairs, reverse_pairs], axis=1), dtype=torch.long)

    if "weight" in usable_edges:
        weights = usable_edges["weight"].to_numpy(dtype=np.float32)
        edge_weight = torch.as_tensor(np.concatenate([weights, weights]), dtype=torch.float32)
    else:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    data = Data(
        x=torch.as_tensor(features.to_numpy(), dtype=torch.float32),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    data.stock_ids = list(stock_ids)
    return data


def build_graph_snapshot(*_args: object, spec: GraphSpec | None = None, **_kwargs: object) -> object:
    """Validate settings; full rolling graph snapshots are reserved for graph modeling."""

    validate_graph_spec(spec or GraphSpec())
    raise NotImplementedError("Rolling graph snapshots are reserved for the graph modeling stage.")

