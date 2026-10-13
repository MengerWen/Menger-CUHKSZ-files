# Stage 8 Final Summary

## Main Finding

The core research question is whether adding graph structure to a conditional latent factor model improves dynamic exposure estimation enough to beat characteristic-only models in out-of-sample pricing, prediction, and portfolio performance.

On the current default `features500` main specification, the strongest evidence is that graph structure adds **incremental cross-sectional structure and economic value**, but not yet a clean all-metric victory.

- Stage 6 graph-model rank IC: 0.0374
- Stage 6 graph-model cross-sectional correlation: 0.0431
- Stage 6 graph-model OOS R^2: -0.0111
- Stage 7 graph-model long-only value-weight annualized return at 10 bps: 0.2098
- Stage 7 graph-model long-only value-weight Sharpe at 10 bps: 1.1794
- Stage 7 graph-model long-short equal-weight annualized return at 10 bps: 0.0732
- Stage 7 graph-model long-short equal-weight Sharpe at 10 bps: 0.6536

## Benchmark Comparison

Relative to the characteristic-only nonlinear latent benchmark (`conditional_autoencoder_style`), the graph model improves rank IC (0.0374 vs -0.0008) and cross-sectional correlation (0.0431 vs 0.0022).

Relative to `ipca_style`, the graph model slightly improves ranking (0.0374 vs 0.0354) but not OOS R^2 (-0.0111 vs -0.0071).

Relative to the direct `mlp_predictor`, the graph model has weaker rank IC (0.0374 vs 0.0582) but materially better OOS R^2 (-0.0111 vs -0.0769) and a much stronger asset-pricing interpretation.

## Economic Value

The clearest economic win is in the long-only tests. The current graph model produces the best long-only Sharpe and return among the four implemented main-spec models. In long-short portfolios, the graph model is competitive on return, but the IPCA-style benchmark remains hard to beat on Sharpe.

That pattern suggests the graph-enhanced beta function is helping identify better top-ranked names for implementable portfolios, while the risk control side of the long-short book still has room to improve.

## Interpretation

The interpretation work in this stage stays close to the models we actually implemented.

Top feature-to-exposure links from the saved latent outputs:

- `conditional_autoencoder_style`: `net_profit_to_asset_ttm` (avg abs monthly Spearman 0.445)
- `conditional_autoencoder_style`: `aqr_quality` (avg abs monthly Spearman 0.438)
- `conditional_autoencoder_style`: `ebit_to_asset_ttm` (avg abs monthly Spearman 0.430)
- `graph_conditional_latent_factor`: `abnormal_turnover_21` (avg abs monthly Spearman 0.315)
- `graph_conditional_latent_factor`: `total_asset_to_total_mktcap_mrq` (avg abs monthly Spearman 0.276)
- `graph_conditional_latent_factor`: `max_plus_21` (avg abs monthly Spearman 0.275)
- `ipca_style`: `fp_score` (avg abs monthly Spearman 0.473)
- `ipca_style`: `west_eps_fy1` (avg abs monthly Spearman 0.464)
- `ipca_style`: `price_to_high_252` (avg abs monthly Spearman 0.461)

Focused graph-model permutation importance from the Stage 8 diagnostic rerun:

- `altman_zscore`: rank IC drop 0.0059
- `abnormal_turnover_21`: rank IC drop 0.0047
- `total_liab_to_total_mktcap_mrq`: rank IC drop 0.0038
- `contributed_capital_to_total_mktcap_mrq`: rank IC drop 0.0037
- `total_profit_to_total_liab_mra`: rank IC drop 0.0036

Graph-neighborhood summaries show that top-decile graph picks sit in denser parts of the monthly graph than the average stock (mean degree top 36.35 vs universe 35.28).

A GAT robustness run was also completed, so attention-weight summaries are available as exploratory evidence.

## Robustness

Focused graph robustness checks were run on a small number of variants that stay close to the implemented pipeline.

- `graph_main_diagnostic`: OOS R^2 -0.0111, rank IC 0.0374, CS corr 0.0431, pricing RMSE 0.0488
- `graph_return_only`: OOS R^2 -0.0069, rank IC 0.0230, CS corr 0.0245, pricing RMSE 0.0483
- `graph_conditional_latent_factor_static_test_graph`: OOS R^2 -0.0452, rank IC 0.0017, CS corr 0.0021, pricing RMSE 0.0496
- `graph_latent_k5`: OOS R^2 -0.0150, rank IC -0.0087, CS corr -0.0002, pricing RMSE 0.0489
- `graph_lookback6`: OOS R^2 -0.0054, rank IC -0.0170, CS corr -0.0145, pricing RMSE 0.0480
- `graph_gat_hybrid`: OOS R^2 -0.0115, rank IC -0.0466, CS corr -0.0208, pricing RMSE 0.0486

Selected portfolio robustness diagnostics:

- `graph_main_diagnostic`: long-only value Sharpe 1.1794, long-short equal Sharpe 0.6536
- `graph_return_only`: long-only value Sharpe 0.8576, long-short equal Sharpe 0.5070
- `graph_latent_k5`: long-only value Sharpe 0.3493, long-short equal Sharpe -0.3712
- `graph_lookback6`: long-only value Sharpe 0.2669, long-short equal Sharpe -1.0290
- `graph_conditional_latent_factor_static_test_graph`: long-only value Sharpe 0.1649, long-short equal Sharpe -0.5471
- `graph_gat_hybrid`: long-only value Sharpe -0.1400, long-short equal Sharpe -1.2869

Implemented robustness variants:

- `graph_return_only`: Return-correlation-only graph
- `graph_lookback6`: Hybrid graph with 6-month return lookback
- `graph_latent_k5`: Hybrid graph with latent dimension K=5
- `graph_gat_hybrid`: Hybrid graph with GAT encoder

## Strong Evidence

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

- `outputs/stage8/plots/stage8_feature_exposure_heatmap.png`
- `outputs/stage8/plots/stage8_permutation_importance.png`
- `outputs/stage8/plots/stage8_neighbor_summary.png`
- `outputs/stage8/plots/stage8_neighbor_edge_mix.png`
- `outputs/stage8/plots/stage8_robustness_prediction_bars.png`
- `outputs/stage8/plots/stage8_robustness_portfolio_bars.png`
- `outputs/stage8/plots/stage8_gat_attention_edge_types.png`
