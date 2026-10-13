# Stage 4 Graph Construction Design

## Data Feasibility Audit

Stage 4 inspected the cleaned Stage 2 panel and the raw files under `Final Project/data/` for explicit industry or sector labels.

Confirmed available graph inputs:

- Cleaned monthly `features500` panel: `outputs/panels/main_features500_panel.pkl`
- Stage 2 feature metadata: `outputs/metadata/main_features500_panel_metadata.json`
- Monthly stock returns: `data/monthly_returns.pkl`
- Month-`t` market capitalization in the cleaned panel: `mcap_t`
- Month-`t` cleaned characteristics, including `amount_21` as an optional liquidity proxy

Confirmed unavailable in the stored dataset:

- No panel columns containing `industry`, `sector`, `classification`, `citic`, `sw`, `申万`, or `中信`
- No feature files in `features500/` or `features/` matching those industry/sector terms
- No matching columns in `price.h5`
- No top-level raw industry classification file under `data/`

Therefore, explicit static industry edges are not implemented in Stage 4. The code leaves an interface for adding an industry edge type later, but the current main graph is similarity-based only.

## Default Main Specification

The default graph is a dynamic hybrid similarity graph for the `features500` main universe.

For each graph month `t`, nodes are the stocks present in the cleaned Stage 2 panel for month `t`. Node features are the 218 cleaned, winsorized, imputed, and cross-sectionally normalized characteristics from Stage 2.

The default edge layers are:

- `return_correlation`: positive return-correlation kNN edges using the past 12 monthly returns through month `t`.
- `feature_cosine_knn`: cosine-similarity kNN edges using month-`t` characteristics.
- `feature_euclidean_knn`: Euclidean kNN edges using month-`t` normalized characteristic vectors, with weights `1 / (1 + distance)`.

These typed edge layers are also combined into a first-pass homogeneous graph by averaging duplicate edge weights across edge types. The typed edge DataFrame is preserved, so later stages can still recover whether an edge came from return correlation, feature similarity, Euclidean feature proximity, or multiple sources.

## Look-Ahead Control

The graph for month `t` uses only information available at or before month `t`:

- Return-correlation edges use `monthly_returns.pkl` over `[t - 11 months, t]` by default.
- Feature-based edges use month-`t` cleaned characteristics from the Stage 2 panel.
- Optional market-cap and liquidity filters use month-`t` panel columns only.
- No `t+1` realized return target is used when constructing graph edges.

The current design assumes graphs are formed after month-end `t` information is available and are then used to predict or price month `t+1` returns. If a stricter signal-timing convention is needed later, set `graph.include_current_month_return: false` to make return-correlation edges use returns only through `t-1`.

## Sparsification and Edge Weights

Default sparsification rules in `configs/graphs_features500.yaml`:

- `return_lookback_months: 12`
- `min_return_observations: 6`
- `k_return: 10`
- `k_feature_cosine: 10`
- `k_feature_euclidean: 10`
- `min_edge_weight: 0.0`
- `combine_rule: mean`

Return-correlation edges keep only positive pairwise correlations after the minimum observation rule. Feature cosine edges use `1 - cosine_distance` as the weight. Feature Euclidean edges use `1 / (1 + distance)` as the weight. Edges are undirected and deduplicated within each layer.

## Saved Artifacts

The pipeline writes month-by-month graph files under:

```text
outputs/graphs/features500_similarity_hybrid/
```

Each month has:

- `edges/YYYY-MM-DD_edges.pkl`: a payload containing typed edges, combined edges, node features, graph date, and config.
- `pyg/YYYY-MM-DD.pt`: a `torch_geometric.data.Data` object using the combined homogeneous edge graph.

Run-level artifacts:

- `outputs/graphs/features500_similarity_hybrid_manifest.csv`
- `outputs/graphs/features500_similarity_hybrid_stats.csv`
- `outputs/metadata/stage4_graph_metadata.json`

The PyG object contains:

- `x`: month-`t` node feature matrix
- `edge_index`: bidirectional homogeneous graph edges
- `edge_weight`: duplicated edge weights for both directions
- `stock_ids`: node order for mapping predictions back to stock identifiers
- `date`, `edge_types`, and `combine_rule` metadata

## Descriptive Graph Stats

The full Stage 4 run built 211 monthly graph snapshots from `2007-01-31` through `2024-07-31`.

Summary of undirected edge counts by month:

| edge layer | months | mean edges | min edges | median edges | max edges |
|---|---:|---:|---:|---:|---:|
| combined | 211 | 7444.77 | 3703 | 7830 | 8903 |
| return_correlation | 211 | 3116.51 | 1585 | 3199 | 3838 |
| feature_cosine_knn | 211 | 2976.04 | 1486 | 3145 | 3489 |
| feature_euclidean_knn | 211 | 3758.32 | 1927 | 3920 | 4366 |

For the final graph month `2024-07-31`, the saved edge payload has 497 nodes, 11,391 typed edges, 8,479 combined undirected edges, and a 497 by 218 node feature matrix. The corresponding PyG object has `edge_index` shape `(2, 16958)` because undirected edges are stored bidirectionally.

## Optional Future Extensions

Implemented now:

- Return-correlation graph from historical returns
- Feature cosine kNN graph from month-`t` characteristics
- Feature Euclidean kNN graph from normalized month-`t` characteristics
- Optional month-`t` market-cap and liquidity node filters
- Month-by-month edge payloads and PyG graph objects

Not implemented because the data is unavailable:

- Static industry graph
- Hybrid static industry plus dynamic similarity graph

Future extension path:

- Add a point-in-time industry classification file with `date`, `stock_id`, and `industry` columns.
- Enable an `industry` edge layer only after confirming the file's timing and schema.
- Preserve the current typed-edge interface so industry edges can be combined with the dynamic similarity graph without changing downstream GNN code.

