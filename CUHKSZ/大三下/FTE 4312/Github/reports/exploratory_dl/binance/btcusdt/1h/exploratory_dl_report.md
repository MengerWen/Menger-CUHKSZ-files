# Exploratory DL Showcase Report

Exploratory DL results are supplementary showcase results designed to demonstrate model learning behavior, ranking ability, and alternative opportunity definitions. They do not replace the strict post-cost primary conclusion.

## Strict Context

- Strict best baseline: `elastic_net_regression` with metric `0.6772883477693387` and signal count `0`
- Strict best deep learning: `transformer_encoder` with metric `0.6460854810538473` and signal count `0`
- Strict backtest verdict: `completed` from `spread_zscore_1p5`

## What Changed In The Exploratory Track

- Exploratory models use independent showcase targets such as gross opportunity and direction-aware opportunity labels.
- Signal generation uses ranking-based and support-aware threshold rules instead of only `predicted_value >= 0`.
- A dedicated exploratory backtest evaluates only exploratory DL signals and keeps strict outputs untouched.

## Showcase Leaderboard

| strategy_name                                        | model_name          | target_type                  | signal_rule                       | evaluation_split   |   trade_count |   cumulative_return |   mark_to_market_max_drawdown |   sharpe_ratio |   total_net_pnl_usd | status    |   reason |
|:-----------------------------------------------------|:--------------------|:-----------------------------|:----------------------------------|:-------------------|--------------:|--------------------:|------------------------------:|---------------:|--------------------:|:----------|---------:|
| lstm_gross__rolling_top_decile_abs                   | lstm                | gross_opportunity_regression | rolling_top_decile_abs            | test               |           578 |           -0.187913 |                     -0.187913 |       -24.1796 |            -18791.3 | completed |      nan |
| gru_gross__rolling_top_decile_abs                    | gru                 | gross_opportunity_regression | rolling_top_decile_abs            | test               |           601 |           -0.195558 |                     -0.195558 |       -24.7342 |            -19555.8 | completed |      nan |
| lstm_direction__rolling_top_decile_abs               | lstm                | direction_classification     | rolling_top_decile_abs            | test               |           645 |           -0.210871 |                     -0.210871 |       -25.7488 |            -21087.1 | completed |      nan |
| tcn_gross__rolling_top_decile_abs                    | tcn                 | gross_opportunity_regression | rolling_top_decile_abs            | test               |           649 |           -0.211737 |                     -0.211737 |       -25.8051 |            -21173.7 | completed |      nan |
| transformer_gross__rolling_top_decile_abs            | transformer_encoder | gross_opportunity_regression | rolling_top_decile_abs            | test               |           651 |           -0.211804 |                     -0.211804 |       -25.8807 |            -21180.4 | completed |      nan |
| gru_direction__rolling_top_decile_abs                | gru                 | direction_classification     | rolling_top_decile_abs            | test               |           661 |           -0.215534 |                     -0.215534 |       -26.081  |            -21553.4 | completed |      nan |
| tcn_direction__rolling_top_decile_abs                | tcn                 | direction_classification     | rolling_top_decile_abs            | test               |           662 |           -0.216053 |                     -0.216053 |       -26.1178 |            -21605.3 | completed |      nan |
| transformer_direction__rolling_top_decile_abs        | transformer_encoder | direction_classification     | rolling_top_decile_abs            | test               |           726 |           -0.237273 |                     -0.237273 |       -27.5217 |            -23727.3 | completed |      nan |
| lstm_gross__validation_tuned_balanced_support        | lstm                | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1249 |           -0.407382 |                     -0.407382 |       -36.4779 |            -40738.2 | completed |      nan |
| gru_gross__validation_tuned_balanced_support         | gru                 | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1279 |           -0.417882 |                     -0.417882 |       -36.982  |            -41788.2 | completed |      nan |
| tcn_gross__validation_tuned_balanced_support         | tcn                 | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1307 |           -0.427849 |                     -0.427849 |       -37.4403 |            -42784.9 | completed |      nan |
| transformer_gross__validation_tuned_balanced_support | transformer_encoder | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1552 |           -0.508434 |                     -0.508434 |       -41.1969 |            -50843.4 | completed |      nan |

## Full Exploratory Leaderboard

| strategy_name                                            | model_name          | target_type                  | signal_rule                       | evaluation_split   |   trade_count |   cumulative_return |   mark_to_market_max_drawdown |   sharpe_ratio |   total_net_pnl_usd | status    |   reason |
|:---------------------------------------------------------|:--------------------|:-----------------------------|:----------------------------------|:-------------------|--------------:|--------------------:|------------------------------:|---------------:|--------------------:|:----------|---------:|
| lstm_gross__rolling_top_decile_abs                       | lstm                | gross_opportunity_regression | rolling_top_decile_abs            | test               |           578 |           -0.187913 |                     -0.187913 |       -24.1796 |            -18791.3 | completed |      nan |
| gru_gross__rolling_top_decile_abs                        | gru                 | gross_opportunity_regression | rolling_top_decile_abs            | test               |           601 |           -0.195558 |                     -0.195558 |       -24.7342 |            -19555.8 | completed |      nan |
| lstm_direction__rolling_top_decile_abs                   | lstm                | direction_classification     | rolling_top_decile_abs            | test               |           645 |           -0.210871 |                     -0.210871 |       -25.7488 |            -21087.1 | completed |      nan |
| tcn_gross__rolling_top_decile_abs                        | tcn                 | gross_opportunity_regression | rolling_top_decile_abs            | test               |           649 |           -0.211737 |                     -0.211737 |       -25.8051 |            -21173.7 | completed |      nan |
| transformer_gross__rolling_top_decile_abs                | transformer_encoder | gross_opportunity_regression | rolling_top_decile_abs            | test               |           651 |           -0.211804 |                     -0.211804 |       -25.8807 |            -21180.4 | completed |      nan |
| gru_direction__rolling_top_decile_abs                    | gru                 | direction_classification     | rolling_top_decile_abs            | test               |           661 |           -0.215534 |                     -0.215534 |       -26.081  |            -21553.4 | completed |      nan |
| tcn_direction__rolling_top_decile_abs                    | tcn                 | direction_classification     | rolling_top_decile_abs            | test               |           662 |           -0.216053 |                     -0.216053 |       -26.1178 |            -21605.3 | completed |      nan |
| transformer_direction__rolling_top_decile_abs            | transformer_encoder | direction_classification     | rolling_top_decile_abs            | test               |           726 |           -0.237273 |                     -0.237273 |       -27.5217 |            -23727.3 | completed |      nan |
| lstm_gross__validation_tuned_balanced_support            | lstm                | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1249 |           -0.407382 |                     -0.407382 |       -36.4779 |            -40738.2 | completed |      nan |
| gru_gross__validation_tuned_balanced_support             | gru                 | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1279 |           -0.417882 |                     -0.417882 |       -36.982  |            -41788.2 | completed |      nan |
| tcn_gross__validation_tuned_balanced_support             | tcn                 | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1307 |           -0.427849 |                     -0.427849 |       -37.4403 |            -42784.9 | completed |      nan |
| transformer_gross__validation_tuned_balanced_support     | transformer_encoder | gross_opportunity_regression | validation_tuned_balanced_support | test               |          1552 |           -0.508434 |                     -0.508434 |       -41.1969 |            -50843.4 | completed |      nan |
| gru_direction__validation_tuned_balanced_support         | gru                 | direction_classification     | validation_tuned_balanced_support | test               |          1835 |           -0.603037 |                     -0.603063 |       -44.7285 |            -60303.7 | completed |      nan |
| lstm_direction__validation_tuned_balanced_support        | lstm                | direction_classification     | validation_tuned_balanced_support | test               |          1881 |           -0.618304 |                     -0.618313 |       -45.1555 |            -61830.4 | completed |      nan |
| tcn_direction__validation_tuned_balanced_support         | tcn                 | direction_classification     | validation_tuned_balanced_support | test               |          1905 |           -0.628009 |                     -0.628018 |       -45.5328 |            -62800.9 | completed |      nan |
| transformer_direction__validation_tuned_balanced_support | transformer_encoder | direction_classification     | validation_tuned_balanced_support | test               |          2021 |           -0.667312 |                     -0.667312 |       -46.8477 |            -66731.2 | completed |      nan |

## Model And Target Comparison

| track       | model_name          | task           | target_column                                           | status   |   test_pearson_corr |   test_rmse |   test_signal_count |    test_f1 |
|:------------|:--------------------|:---------------|:--------------------------------------------------------|:---------|--------------------:|------------:|--------------------:|-----------:|
| strict      | lstm                | regression     | target_future_net_return_bps_24h                        | warning  |            0.638585 |     1.23896 |                   0 | nan        |
| strict      | gru                 | regression     | target_future_net_return_bps_24h                        | warning  |            0.570399 |     1.36035 |                   0 | nan        |
| strict      | tcn                 | regression     | target_future_net_return_bps_24h                        | warning  |            0.474455 |     1.77152 |                   0 | nan        |
| strict      | transformer_encoder | regression     | target_future_net_return_bps_24h                        | warning  |            0.646085 |     1.20869 |                   0 | nan        |
| exploratory | lstm                | regression     | target_future_short_perp_long_spot_gross_return_bps_24h | ok       |            0.635403 |     1.25115 |                 188 | nan        |
| exploratory | gru                 | regression     | target_future_short_perp_long_spot_gross_return_bps_24h | ok       |            0.627587 |     1.2318  |                 145 | nan        |
| exploratory | tcn                 | regression     | target_future_short_perp_long_spot_gross_return_bps_24h | ok       |            0.624854 |     1.26268 |                 186 | nan        |
| exploratory | transformer_encoder | regression     | target_future_short_perp_long_spot_gross_return_bps_24h | ok       |            0.644495 |     1.21687 |                 102 | nan        |
| exploratory | lstm                | classification | target_best_direction_is_short_perp_long_spot_24h       | ok       |          nan        |   nan       |                 891 |   0.256683 |
| exploratory | gru                 | classification | target_best_direction_is_short_perp_long_spot_24h       | ok       |          nan        |   nan       |                1092 |   0.306264 |
| exploratory | tcn                 | classification | target_best_direction_is_short_perp_long_spot_24h       | ok       |          nan        |   nan       |                1017 |   0.286438 |
| exploratory | transformer_encoder | classification | target_best_direction_is_short_perp_long_spot_24h       | ok       |          nan        |   nan       |                1692 |   0.431144 |

## Direction Summary

| direction            |   trade_count |   total_net_pnl_usd |   average_net_return_bps |
|:---------------------|--------------:|--------------------:|-------------------------:|
| long_perp_short_spot |         34556 |        -1.11407e+06 |                 -32.2395 |
| short_perp_long_spot |         55398 |        -1.76472e+06 |                 -31.8554 |

## Top-Quantile Diagnostic Summary

| run_name              |   top_bucket_row_count |   top_bucket_avg_directional_return_bps |   top_bucket_cumulative_directional_return_bps |
|:----------------------|-----------------------:|----------------------------------------:|-----------------------------------------------:|
| lstm_gross            |                    883 |                                 2.31691 |                                        2045.83 |
| gru_gross             |                    883 |                                 2.31992 |                                        2048.49 |
| tcn_gross             |                    883 |                                 2.30629 |                                        2036.45 |
| transformer_gross     |                    883 |                                 2.37213 |                                        2094.59 |
| lstm_direction        |                    883 |                                 1.44193 |                                        1273.22 |
| gru_direction         |                    883 |                                 1.92905 |                                        1703.35 |
| tcn_direction         |                    883 |                                 1.64061 |                                        1448.66 |
| transformer_direction |                    883 |                                 1.87512 |                                        1655.73 |

## Files

- Full leaderboard: `D:\MG\! CUHKSZ\~！大三 下\FTE 4312\Github\reports\exploratory_dl\binance\btcusdt\1h\exploratory_full_leaderboard.csv`
- Showcase leaderboard: `D:\MG\! CUHKSZ\~！大三 下\FTE 4312\Github\reports\exploratory_dl\binance\btcusdt\1h\exploratory_showcase_leaderboard.json`
- Prediction distribution: `D:\MG\! CUHKSZ\~！大三 下\FTE 4312\Github\reports\exploratory_dl\binance\btcusdt\1h\exploratory_prediction_distribution.json`
- Quantile analysis: `D:\MG\! CUHKSZ\~！大三 下\FTE 4312\Github\reports\exploratory_dl\binance\btcusdt\1h\exploratory_quantile_analysis.json`