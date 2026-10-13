# Industry Classification Extension

## Scope

This extension adds the teacher-provided static first-level industry classification `data/ind_code.pkl` as an additional graph-relation source. It does not overwrite the original Stage 3-8 outputs. Industry codes are used only to form graph edges, not as extra model features.

Because the file has no date dimension, the industry relation is treated as a static industry prior rather than a point-in-time changing industry history.

## Industry Data Audit

- Label file: `data/ind_code.pkl`
- Label name: `industry_code`
- Covered panel stocks: 1676 / 1678 (99.88%)
- Covered panel rows: 90322 / 90351 (99.97%)
- Monthly industry count range: 28 to 30
- Average same-industry clique edges per month: 4380.15

## Added Model Variants

- `graph_industry_only`: uses only same-industry edges.
- `graph_industry_hybrid`: uses the original dynamic similarity layers plus same-industry edges.

## Prediction and Pricing Comparison

| model | oos_r2_zero_benchmark | rank_ic_mean | cross_sectional_corr_mean | pricing_error_monthly_rmse |
| --- | --- | --- | --- | --- |
| graph_conditional_latent_factor | -0.0111 | 0.0374 | 0.0431 | 0.0488 |
| graph_industry_hybrid | -0.0069 | 0.0005 | -0.0012 | 0.0482 |
| graph_industry_only | -0.0015 | 0.0144 | 0.0224 | 0.0478 |

## Portfolio Comparison

The portfolio table uses the main transaction-cost setting of `10` bps.

| model | long_only_value_ann_return | long_only_value_sharpe | long_short_equal_ann_return | long_short_equal_sharpe |
| --- | --- | --- | --- | --- |
| graph_conditional_latent_factor | 0.2098 | 1.1794 | 0.0732 | 0.6536 |
| graph_industry_hybrid | 0.0752 | 0.3835 | -0.0466 | -0.3212 |
| graph_industry_only | 0.0724 | 0.3298 | 0.0024 | 0.0749 |

## Figures

- `figure_9_industry_graph_extension`: `reports/figures/figure_9_industry_graph_extension.png`
- `figure_10_industry_portfolio_extension`: `reports/figures/figure_10_industry_portfolio_extension.png`

## Interpretation Guide

If `graph_industry_hybrid` improves on the original graph model, the static industry relation is adding useful structure beyond dynamic similarity. If `graph_industry_only` is weaker but `graph_industry_hybrid` is competitive, industry is best read as a stable prior rather than a complete substitute for dynamic graph information. If neither industry variant improves the result, the original similarity-hybrid graph remains valuable because it is not merely recreating broad industry clusters.

## Limitations

1. The industry classification is static, so it cannot capture historical industry reclassifications.
2. The extension keeps the same short 24-month OOS window as the original main comparison.
3. The graph encoder is still the same compact GCN latent-factor model; this extension changes relation structure, not the core pricing architecture.
