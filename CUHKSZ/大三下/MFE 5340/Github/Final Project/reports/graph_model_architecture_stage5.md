# Stage 5 Graph-Enhanced Conditional Latent Factor Model

## Research Role

Stage 5 implements the first graph-enhanced conditional latent factor model for the project. The model is not intended to be a pure graph return predictor. Its economic role is to test whether graph context improves the estimation of dynamic exposures:

```text
beta_{i,t} = g_theta(X_t, G_t)
y_hat_{i,t+1} = beta_{i,t}' f_bar
```

where `X_t` is the month-`t` cleaned characteristic matrix, `G_t` is the month-`t` stock graph from Stage 4, and `f_bar` is the historical mean latent factor premium learned from the training window.

## Architecture

The default architecture is a compact two-layer GCN beta encoder:

```text
X_t, edge_index_t, edge_weight_t
  -> GCN layer
  -> ReLU + dropout
  -> GCN layer
  -> beta_{i,t} in R^K
```

The model also supports a GAT beta encoder through config (`model.model_type: gat`). When GAT is used, final-layer attention weights can be exported to `outputs/attention/stage5_graph_attention.pkl`. The default run uses GCN to keep the first graph specification simple and defensible.

Latent factors are represented as train-month factor embeddings. During training, the model reconstructs next-month excess returns using:

```text
y_{i,t+1} ~= beta_{i,t}' f_t
```

For validation and OOS prediction, it uses:

```text
y_hat_{i,t+1} = beta_{i,t}' mean_t(f_t)
```

This mirrors the Stage 3 CAE-style benchmark's idea of nonlinear characteristic-driven beta, but replaces the characteristic-only beta network with a graph-aware beta encoder.

## Loss Components

The Stage 5 config supports four loss terms:

- `reconstruction_loss_weight`: MSE of `beta_{i,t}' f_t` on train-month latent factors.
- `prediction_loss_weight`: MSE of `beta_{i,t}' mean(f_train)` to align training with the OOS prediction rule.
- `pricing_error_weight`: squared cross-sectional mean residual within each month.
- `beta_l2_weight`: small exposure shrinkage penalty.

The default model is still a simple asset-pricing-oriented neural benchmark. It is not a full no-arbitrage SDF system.

## Graph Inputs

Stage 5 consumes the Stage 4 PyG graph snapshots from:

```text
outputs/graphs/features500_similarity_hybrid_manifest.csv
outputs/graphs/features500_similarity_hybrid/pyg/YYYY-MM-DD.pt
```

The graph for month `t` contains:

- `x`: cleaned month-`t` node characteristics
- `edge_index`: combined bidirectional graph edges
- `edge_weight`: combined graph edge weights
- `stock_ids`: node order metadata

The current graph is a dynamic hybrid similarity graph built from return correlation, feature cosine kNN, and feature Euclidean kNN. Explicit industry edges are not used because Stage 4 confirmed that industry labels are unavailable in the current stored dataset.

## Look-Ahead Control

For each OOS refit block:

- Training graphs use months up to the training end date.
- Validation graphs use the trailing pre-test validation window.
- Test graphs use only month-`t` graph snapshots and month-`t` node features to predict month `t+1` excess returns.
- No `t+1` realized return is used in graph construction or beta estimation at prediction time.

The return-correlation graph layer uses returns through month `t`, consistent with a month-end signal formed after month `t` is observed.

## Difference From CAE-Style Benchmark

The Stage 3 CAE-style model learns:

```text
beta_{i,t} = g_theta(x_{i,t})
```

Stage 5 learns:

```text
beta_{i,t} = g_theta(X_t, G_t)_i
```

The difference is where information enters the exposure function. CAE-style beta depends only on the stock's own characteristics. Stage 5 beta depends on the stock's own characteristics and messages from graph neighbors, so the exposure can reflect dynamic similarity, co-movement, and local cross-sectional context.

## Simplifications

This first graph model is intentionally simpler than an ideal structural asset-pricing model:

- No explicit no-arbitrage SDF constraint yet.
- No separate macro state encoder.
- No multi-relation GNN; Stage 4 typed edges are combined into one homogeneous graph for the default PyG input.
- No industry edges because the dataset does not contain point-in-time industry labels.
- Latent factor forecasts use the historical mean train factor embedding rather than a separate factor forecasting model.
- Factor signs and rotations remain unidentified, as in latent factor models generally.

These simplifications keep Stage 5 focused on the core empirical question: does graph context improve conditional exposure learning beyond characteristic-only nonlinear beta models?
