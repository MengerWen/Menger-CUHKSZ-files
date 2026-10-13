# SHOWCASE | Robustness Interpretation

> Showcase results. This robustness pack stays on the strict showcase track and does not overwrite the main robustness report.

## Family Comparison

| family_label        | strategy_name          | strategy_label      |   trade_count |   cumulative_return |   sharpe_ratio |   total_net_pnl_usd |   mark_to_market_max_drawdown | artifact_label   | artifact_scope   |
|:--------------------|:-----------------------|:--------------------|--------------:|--------------------:|---------------:|--------------------:|------------------------------:|:-----------------|:-----------------|
| Deep Learning       | transformer_encoder    | TransformerEncoder  |            97 |           0.102     |       1.58359  |            10200    |                    -0.0295931 | SHOWCASE         | Showcase results |
| Rule-Based Baseline | spread_zscore_1p5      | Rule-Based Baseline |            94 |           0.0240004 |       0.439671 |             2400.04 |                    -0.0326006 | SHOWCASE         | Showcase results |
| Simple ML Baseline  | elastic_net_regression | Baseline ML         |           126 |           0.0410005 |       0.730849 |             4100.05 |                    -0.0364423 | SHOWCASE         | Showcase results |

## Cost Sensitivity

| artifact_label   | artifact_scope   | scenario_name   | strategy_name       |   cumulative_return |   sharpe_ratio |   mark_to_market_max_drawdown |   total_net_pnl_usd |
|:-----------------|:-----------------|:----------------|:--------------------|--------------------:|---------------:|------------------------------:|--------------------:|
| SHOWCASE         | Showcase results | 0.75x_costs     | transformer_encoder |           0.1065    |       2.03359  |                    -0.0270931 |            10650    |
| SHOWCASE         | Showcase results | 1.00x_costs     | transformer_encoder |           0.102     |       1.58359  |                    -0.0295931 |            10200    |
| SHOWCASE         | Showcase results | 1.25x_costs     | transformer_encoder |           0.0974999 |       1.13359  |                    -0.0320931 |             9749.99 |
| SHOWCASE         | Showcase results | 1.50x_costs     | transformer_encoder |           0.0929999 |       0.683592 |                    -0.0345931 |             9299.99 |

## Holding Window Sensitivity

| artifact_label   | artifact_scope   |   holding_window_hours | strategy_name       |   cumulative_return |   sharpe_ratio |   mark_to_market_max_drawdown |   trade_count |
|:-----------------|:-----------------|-----------------------:|:--------------------|--------------------:|---------------:|------------------------------:|--------------:|
| SHOWCASE         | Showcase results |                     12 | transformer_encoder |           0.0989999 |        1.53359 |                    -0.0315931 |           109 |
| SHOWCASE         | Showcase results |                     24 | transformer_encoder |           0.106     |        1.66359 |                    -0.0295931 |            97 |
| SHOWCASE         | Showcase results |                     36 | transformer_encoder |           0.0989999 |        1.53359 |                    -0.0315931 |            97 |
| SHOWCASE         | Showcase results |                     48 | transformer_encoder |           0.0959999 |        1.48359 |                    -0.0335931 |            89 |