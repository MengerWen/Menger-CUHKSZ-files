# Deep-Learning Comparison Report

- Experiment: `unit_sequence_regression`
- Description: unit test bundle
- Compared runs: `2`
- Validation ranking metric: `pearson_corr`
- Test ranking metric: `pearson_corr`
- Strategy ranking metric: `top_quantile_avg_return_bps` on `test`

## Runs

- `lstm` -> `lstm` using `D:\MG\! CUHKSZ\~！大三 下\FTE 4312\Github\tests\.tmp\deep_learning_experiments\reuse_artifacts\configs\lstm.yaml` (reused_artifacts=True)
- `gru` -> `gru` using `D:\MG\! CUHKSZ\~！大三 下\FTE 4312\Github\tests\.tmp\deep_learning_experiments\reuse_artifacts\configs\gru.yaml` (reused_artifacts=True)

## Main Finding

Current default best model under the configured test ranking metric (`pearson_corr`) is `lstm` with score `0.6`.

## Validation Leaderboard

|   rank | run_label   | model_name   | model_group   | status   | reason   | degenerate_experiment   | task       | target_column                    |   lookback_steps |   best_epoch | selected_loss   | checkpoint_selection_metric   | checkpoint_selection_effective_metric   |   selected_threshold | ranking_metric   |   ranking_metric_value |   validation_pearson_corr |   validation_rmse |   validation_avg_signal_return_bps |   validation_cumulative_signal_return_bps |   validation_signal_hit_rate |   validation_signal_count |   validation_top_quantile_avg_return_bps |
|-------:|:------------|:-------------|:--------------|:---------|:---------|:------------------------|:-----------|:---------------------------------|-----------------:|-------------:|:----------------|:------------------------------|:----------------------------------------|---------------------:|:-----------------|-----------------------:|--------------------------:|------------------:|-----------------------------------:|------------------------------------------:|-----------------------------:|--------------------------:|-----------------------------------------:|
|      1 | lstm        | lstm         | recurrent     | ok       |          | False                   | regression | target_future_net_return_bps_24h |                4 |            2 | huber           | validation_pearson_corr       | validation_pearson_corr                 |                    0 | pearson_corr     |                    0.7 |                       0.7 |                 2 |                                  1 |                                        10 |                          0.6 |                         3 |                                        7 |
|      2 | gru         | gru          | recurrent     | ok       |          | False                   | regression | target_future_net_return_bps_24h |                4 |            2 | huber           | validation_pearson_corr       | validation_pearson_corr                 |                    0 | pearson_corr     |                    0.5 |                       0.5 |                 2 |                                  1 |                                        10 |                          0.6 |                         3 |                                        5 |

## Test Leaderboard

|   rank | run_label   | model_name   | model_group   | status   | reason   | degenerate_experiment   | task       | target_column                    |   lookback_steps |   best_epoch | selected_loss   | checkpoint_selection_metric   | checkpoint_selection_effective_metric   |   selected_threshold | ranking_metric   |   ranking_metric_value |   test_pearson_corr |   test_rmse |   test_avg_signal_return_bps |   test_cumulative_signal_return_bps |   test_signal_hit_rate |   test_signal_count |   test_top_quantile_avg_return_bps |
|-------:|:------------|:-------------|:--------------|:---------|:---------|:------------------------|:-----------|:---------------------------------|-----------------:|-------------:|:----------------|:------------------------------|:----------------------------------------|---------------------:|:-----------------|-----------------------:|--------------------:|------------:|-----------------------------:|------------------------------------:|-----------------------:|--------------------:|-----------------------------------:|
|      1 | lstm        | lstm         | recurrent     | ok       |          | False                   | regression | target_future_net_return_bps_24h |                4 |            2 | huber           | validation_pearson_corr       | validation_pearson_corr                 |                    0 | pearson_corr     |                    0.6 |                 0.6 |         2.5 |                            2 |                                  20 |                    0.7 |                   4 |                                  6 |
|      2 | gru         | gru          | recurrent     | ok       |          | False                   | regression | target_future_net_return_bps_24h |                4 |            2 | huber           | validation_pearson_corr       | validation_pearson_corr                 |                    0 | pearson_corr     |                    0.4 |                 0.4 |         2.5 |                            2 |                                  20 |                    0.7 |                   4 |                                  4 |

## Strategy-Oriented Leaderboard

|   rank | run_label   | model_name   | model_group   | status   | reason   | degenerate_experiment   | task       | target_column                    |   lookback_steps |   best_epoch | selected_loss   | checkpoint_selection_metric   | checkpoint_selection_effective_metric   |   selected_threshold | ranking_metric              |   ranking_metric_value |   test_avg_signal_return_bps |   test_cumulative_signal_return_bps |   test_signal_hit_rate |   test_signal_count |   test_top_quantile_avg_return_bps |   test_top_quantile_cumulative_return_bps |
|-------:|:------------|:-------------|:--------------|:---------|:---------|:------------------------|:-----------|:---------------------------------|-----------------:|-------------:|:----------------|:------------------------------|:----------------------------------------|---------------------:|:----------------------------|-----------------------:|-----------------------------:|------------------------------------:|-----------------------:|--------------------:|-----------------------------------:|------------------------------------------:|
|      1 | lstm        | lstm         | recurrent     | ok       |          | False                   | regression | target_future_net_return_bps_24h |                4 |            2 | huber           | validation_pearson_corr       | validation_pearson_corr                 |                    0 | top_quantile_avg_return_bps |                      6 |                            2 |                                  20 |                    0.7 |                   4 |                                  6 |                                        60 |
|      2 | gru         | gru          | recurrent     | ok       |          | False                   | regression | target_future_net_return_bps_24h |                4 |            2 | huber           | validation_pearson_corr       | validation_pearson_corr                 |                    0 | top_quantile_avg_return_bps |                      4 |                            2 |                                  20 |                    0.7 |                   4 |                                  4 |                                        40 |

## Notes

- This Phase 2 comparison layer reuses the stable Phase 1 per-model artifact contract rather than inventing a new tracking system.
- Runs are compared only after validating that provider, symbol, frequency, task, and target column match.
- Artifact reuse is allowed so repeated comparison refreshes stay lightweight when model outputs already exist.
- Comparison tables now preserve per-run `status`, `reason`, and `degenerate_experiment` so a high predictive score cannot silently hide a no-signal trading regime.

## Figures

- `validation_metric_comparison.png`
- `test_metric_comparison.png`
- `strategy_metric_comparison.png`