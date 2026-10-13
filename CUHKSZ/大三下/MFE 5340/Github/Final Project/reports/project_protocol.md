# Project Protocol

## Project Objective

This project studies whether explicit stock relationship structure improves conditional latent factor pricing in the Chinese equity cross-section. The core object of interest is the conditional exposure function, not a standalone return-prediction model:

```text
conditional beta / dynamic exposure = g(firm characteristics, point-in-time graph context)
```

The main empirical question is whether graph context improves time-varying exposure estimates and, as a consequence, out-of-sample pricing, return prediction, and portfolio performance relative to non-graph conditional factor benchmarks.

## Target Variable

The planned baseline target is one-month-ahead stock excess return. The raw return source is expected to be `data/monthly_returns.pkl`, with the risk-free series from `data/risk_free.csv` used for excess-return construction when the date frequency and alignment support it.

Stage 2 must inspect the exact index, column structure, return definition, and risk-free compounding convention before constructing the final target. This Stage 1 protocol does not assume unavailable fields beyond the filenames and the bundled `data/instruction.txt` description.

## Prediction Horizon

The main horizon is one month ahead. At each month-end `t`, features and graph inputs available no later than `t` are used to predict returns over `t + 1`. Multi-month horizons are not part of the main specification unless added later as a robustness extension.

## Point-in-Time Principles

All preprocessing and experiments should respect the following rules:

- Do not use information from month `t + 1` or later when forming signals at month `t`.
- Align firm characteristics, market capitalization, filters, and graph edges by their availability date before joining them to the next-month return target.
- Apply `BLACKLIST.pkl` and `UNTRADABLE.pkl` only if Stage 2 confirms their index semantics and date alignment.
- Use `csi500_mask_monthly.pkl` to define CSI 500 eligibility when running the main specification, unless Stage 2 discovers that `features500/` already applies exactly the same filter and no additional mask is needed.
- Fit scalers, imputers, feature filters, model parameters, graph thresholds, and hyperparameters using only the training and validation windows available at that OOS date.
- Save intermediate panels, graph snapshots, model outputs, and evaluation tables under `outputs/` with configs and seeds for reproducibility.

## Rolling vs Expanding OOS Protocol

The default protocol is expanding-window OOS evaluation:

- Use an initial historical training window.
- Reserve a trailing validation window for model selection within the historical sample.
- Refit or update models at a fixed monthly frequency.
- Generate one-month-ahead predictions for the next OOS month only.
- Append the realized OOS month and repeat.

Rolling-window evaluation is a robustness alternative. It keeps a fixed-length training history and drops the oldest months as the OOS window advances. The same point-in-time feature, graph, and validation rules apply.

## Benchmark List

Main benchmark families:

- IPCA-style model: linear conditional latent factor benchmark using firm characteristics.
- Conditional Autoencoder-style model: nonlinear non-graph conditional latent factor benchmark.
- MLP predictor: nonlinear direct return-prediction benchmark that does not explicitly estimate latent factors or graph structure.
- Graph model with GCN-style propagation: graph-enhanced conditional latent factor model.
- Graph model with GAT-style attention: graph-enhanced conditional latent factor model with attention over neighbors.

The benchmark design should separate gains from nonlinearity from gains due to graph structure.

## Main Evaluation Metrics

Primary metrics:

- OOS R2 for return prediction.
- Rank IC for cross-sectional ranking ability.
- Pricing error for asset-pricing fit.
- Long-short portfolio return, volatility, Sharpe ratio, drawdown, and turnover.
- Long-only portfolio return, volatility, Sharpe ratio, drawdown, and turnover.

Additional diagnostics can include exposure stability, factor return stability, graph ablations, and performance by market regime, but these should not replace the main specification.

## Main-Spec vs Robustness-Spec Universe

Main specification:

- Use `data/features500/` as the default characteristic universe because the local data instruction says it was filtered using the CSI 500 filter at each month-end.
- Stage 1 filesystem inspection found 218 `.pkl` files in `features500/` with filenames matching `features/`, but smaller total byte size, which is consistent with a narrower universe.
- Use `data/csi500_mask_monthly.pkl` to verify the membership filter and to document the exact sample rule in Stage 2.

Robustness specification:

- Use `data/features/` as the broader characteristic universe.
- Treat this as a robustness extension after the main CSI 500 workflow is reproducible.
