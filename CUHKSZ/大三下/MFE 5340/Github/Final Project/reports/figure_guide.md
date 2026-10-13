# Figure Guide

## Existing Plot Inventory

- `stage6_plots`: 4 PNG files found under `outputs/comparison/stage6_plots`.
  - Example: `outputs/comparison/stage6_plots/stage6_monthly_cross_sectional_corr.png`
  - Example: `outputs/comparison/stage6_plots/stage6_monthly_rank_ic.png`
  - Example: `outputs/comparison/stage6_plots/stage6_overall_metric_bars.png`
- `stage7_plots`: 5 PNG files found under `outputs/portfolio/stage7_plots`.
  - Example: `outputs/portfolio/stage7_plots/stage7_cumulative_long_only_equal_10bps.png`
  - Example: `outputs/portfolio/stage7_plots/stage7_cumulative_long_only_value_10bps.png`
  - Example: `outputs/portfolio/stage7_plots/stage7_cumulative_long_short_equal_10bps.png`
- `stage8_plots`: 12 PNG files found under `outputs/stage8/plots`.
  - Example: `outputs/stage8/plots/portfolio_robustness/stage7_cumulative_long_only_equal_10bps.png`
  - Example: `outputs/stage8/plots/portfolio_robustness/stage7_cumulative_long_only_value_10bps.png`
  - Example: `outputs/stage8/plots/portfolio_robustness/stage7_cumulative_long_short_equal_10bps.png`
- `reports`: 8 PNG files found under `reports`.
  - Example: `reports/figures/figure_1_sample_coverage.png`
  - Example: `reports/figures/figure_2_graph_overview.png`
  - Example: `reports/figures/figure_3_model_comparison.png`

## Important Gaps Before This Stage

- Git-tracked report figure package under reports/figures/
- Data and sample overview figure for stock coverage and raw feature missingness.
- Graph construction overview figure for nodes, combined edges, and edge-layer averages.
- A unified figure guide mapping charts to source files and report interpretation.

## Created Report Figures

### Sample Coverage and Raw Feature Missingness

- Figure id: `figure_1_sample_coverage`
- Classification: `main`
- PNG: `reports/figures/figure_1_sample_coverage.png`
- PDF: `reports/figures/figure_1_sample_coverage.pdf`
- Source files:
  - `outputs/panels/main_features500_panel.pkl`
  - `outputs/metadata/main_features500_panel_metadata.json`
- How to read it: The left panel shows the active monthly stock universe in the main-spec panel. The right panel shows which saved raw characteristics were most sparse before cleaning, which helps frame the feature engineering burden behind the benchmark and graph models.

### Graph Construction Overview

- Figure id: `figure_2_graph_overview`
- Classification: `main`
- PNG: `reports/figures/figure_2_graph_overview.png`
- PDF: `reports/figures/figure_2_graph_overview.pdf`
- Source files:
  - `outputs/graphs/features500_similarity_hybrid_manifest.csv`
  - `outputs/graphs/features500_similarity_hybrid_stats.csv`
- How to read it: The left panel shows how the monthly graph size evolves over time. The right panel compares the average sparsified edge count contributed by each implemented edge layer, which makes the hybrid graph design concrete.

### Main Benchmark Comparison

- Figure id: `figure_3_model_comparison`
- Classification: `main`
- PNG: `reports/figures/figure_3_model_comparison.png`
- PDF: `reports/figures/figure_3_model_comparison.pdf`
- Source files:
  - `outputs/stage8/tables/stage8_main_results_table.csv`
- How to read it: This figure is the apples-to-apples benchmark comparison on the aligned main-spec OOS sample. It shows that the graph model improves some cross-sectional metrics relative to characteristic-only latent models, but not every metric simultaneously.

### Cumulative Portfolio Performance

- Figure id: `figure_4_portfolio_cumulative`
- Classification: `main`
- PNG: `reports/figures/figure_4_portfolio_cumulative.png`
- PDF: `reports/figures/figure_4_portfolio_cumulative.pdf`
- Source files:
  - `outputs/portfolio/stage7_monthly_returns.pkl`
- How to read it: These are the investable cumulative return paths on the default main transaction-cost setting. The left panel matches the main long-only result used in the project summary, while the right panel shows whether the graph signal also survives in a market-neutral sorting test.

### Portfolio Return and Sharpe Summary

- Figure id: `figure_5_portfolio_summary`
- Classification: `main`
- PNG: `reports/figures/figure_5_portfolio_summary.png`
- PDF: `reports/figures/figure_5_portfolio_summary.pdf`
- Source files:
  - `outputs/portfolio/stage7_performance_summary.csv`
- How to read it: This summary figure turns the portfolio backtest into clean cross-model comparisons at the main cost setting. It helps separate whether the graph model wins through higher return, better risk control, or both.

### Interpretability and Graph Neighborhood Diagnostics

- Figure id: `figure_6_interpretability`
- Classification: `main`
- PNG: `reports/figures/figure_6_interpretability.png`
- PDF: `reports/figures/figure_6_interpretability.pdf`
- Source files:
  - `outputs/stage8/tables/stage8_feature_exposure_top_links.csv`
  - `outputs/stage8/tables/stage8_permutation_importance.csv`
  - `outputs/stage8/tables/stage8_neighbor_edge_mix.csv`
- How to read it: The heatmap shows which saved features align most strongly with latent exposures across the implemented conditional models. The permutation chart and edge-mix chart then focus on what seems to matter most for the main graph model and what kind of neighborhood structure its top-ranked names sit in.

### Graph Robustness Checks

- Figure id: `figure_7_graph_robustness`
- Classification: `main`
- PNG: `reports/figures/figure_7_graph_robustness.png`
- PDF: `reports/figures/figure_7_graph_robustness.pdf`
- Source files:
  - `outputs/stage8/tables/stage8_graph_robustness_summary.csv`
  - `outputs/stage8/tables/stage8_graph_robustness_portfolio.csv`
- How to read it: This figure keeps the robustness section focused on a small number of strong checks. It shows whether the main hybrid graph remains the best choice once we vary the graph definition, latent dimension, or test-time graph behavior.

### Exploratory GAT Attention Summary

- Figure id: `figure_8_gat_attention_exploratory`
- Classification: `exploratory`
- PNG: `reports/figures/figure_8_gat_attention_exploratory.png`
- PDF: `reports/figures/figure_8_gat_attention_exploratory.pdf`
- Source files:
  - `outputs/stage8/tables/stage8_gat_attention_summary.csv`
- How to read it: This figure is exploratory because the main Stage 5 result is still a GCN run. It shows which saved edge-type combinations receive the largest average attention weights in the successful GAT robustness run.

## Main vs Exploratory

Main result figures are the figures marked `main`. They are the charts that best support the project narrative in the final write-up.

Exploratory figures are supportive diagnostics that are useful in an appendix or discussion section, but should not carry the main claim by themselves.

## Recommended Final Report Set

- `figure_1_sample_coverage`: Sample Coverage and Raw Feature Missingness
- `figure_2_graph_overview`: Graph Construction Overview
- `figure_3_model_comparison`: Main Benchmark Comparison
- `figure_4_portfolio_cumulative`: Cumulative Portfolio Performance
- `figure_5_portfolio_summary`: Portfolio Return and Sharpe Summary
- `figure_7_graph_robustness`: Graph Robustness Checks
