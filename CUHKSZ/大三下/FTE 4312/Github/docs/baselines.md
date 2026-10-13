# Baseline Strategies and Predictive Models

## Purpose

This module is the first serious benchmark layer for the project. Its job is not to "win" the strategy problem by itself, but to answer a more disciplined question:

> Before we trust deep learning, how far can interpretable rules and lightweight supervised models go on a low-signal, post-cost funding arbitrage task?

That makes the baseline layer important for both methodology and presentation:

- it gives simple, explainable reference points
- it exposes how sparse tradable post-cost opportunities really are
- it provides prediction artifacts that later feed unified signal generation and backtesting

## What Changed in the Upgraded Baseline Pipeline

The current baseline module is no longer a simple one-shot fit with fixed thresholds.
It now includes:

- time-series-safe hyperparameter tuning on the `train` split only
- validation-driven threshold selection for rule, classification, and regression baselines
- optional classifier probability calibration
- configurable missing-data handling with safe forward-fill support and missing indicators
- stronger penalized-linear baselines
- held-out permutation importance for final interpretation
- optional expanding / rolling walk-forward prediction mode

The CLI entry points did not change:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main train-baseline --config configs/models/baseline.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main evaluate-baseline --config configs/models/baseline.yaml
```

## Inputs

The baseline pipeline consumes the supervised dataset produced by `build-labels`:

- default input: `data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_dataset.parquet`
- default classification target: `target_is_profitable_24h`
- default regression target: `target_future_net_return_bps_24h`
- default split column: `split`
- default readiness filter: `supervised_ready == 1`

The baseline layer assumes the supervised dataset already contains:

- leakage-safe engineered features
- post-cost labels
- time-series split assignment

## Baseline Families

### Rule-based baselines

Implemented examples:

1. `funding_threshold_2bps`
2. `spread_zscore_1p5`
3. `combined_funding_spread`

These rules remain intentionally interpretable:

- positive funding supports `short perp + long spot`
- positive spread z-score suggests the perp is relatively rich to spot
- the combined rule checks whether carry and basis dislocation line up together

Rules now optionally support validation-set threshold search through configurable grids.

### Predictive classification baselines

Implemented:

1. Logistic regression with L2 penalty
2. Logistic regression with L1 penalty
3. Logistic regression with elastic-net penalty
4. Optional random-forest classifier

Output:

- calibrated or uncalibrated probability of `target_is_profitable_24h == 1`
- validation-selected trade threshold

### Predictive regression baselines

Implemented:

1. Ridge regression
2. ElasticNet regression
3. Optional random-forest regressor

Output:

- predicted future net return in basis points
- validation-selected return threshold for turning forecasts into trade candidates

## Time-Series-Safe Tuning

Hyperparameter tuning is done only inside the `train` split.

The tuning workflow uses chronological inner folds, not shuffled CV. Configurable controls include:

- `tuning.n_splits`
- `tuning.gap`
- `tuning.mode`
  Supported: `expanding`, `rolling`
- `tuning.min_train_size`
- `tuning.rolling_window_size`

This matters because:

- observations are time-ordered
- labels overlap across nearby timestamps
- the project is a trading problem, not an IID tabular benchmark

The default config uses a non-zero `gap` to reduce leakage risk near split boundaries.

## Threshold Selection

The upgraded baseline layer does not rely only on fixed thresholds from config.

Instead, after the model is trained on `train`, it can search thresholds on `validation`:

- classifier probability threshold search
- regression expected-return threshold search
- rule-threshold grid search

The optimization target is configurable through `threshold_search.objective`.

Useful objectives include:

- `avg_signal_return_bps`
- `cumulative_signal_return_bps`
- `precision`
- `f1`
- `signal_hit_rate`
- `signal_sharpe_like`

For this project, `avg_signal_return_bps` is a strong default because the target is trading usefulness, not raw classification accuracy alone.

## Degenerate Threshold Search Handling

Threshold search is now guarded explicitly.

Default behavior:

- if the validation split cannot support threshold selection, the run fails
- if every threshold candidate has an invalid objective or zero usable traded signals, the run fails
- if a model produces zero signals on validation or test, the report and manifest record that status instead of silently presenting a healthy-looking zero row

The baseline artifacts now carry:

- `degenerate_experiment`
- `status`
- `reason`
- `fallback_used`
- `fallback_reason`
- `signal_count_by_split`
- `tradeable_rate_by_split`
- `profitable_rate_by_split`
- `threshold_search_summary`

If you intentionally want to keep writing diagnostic artifacts even when threshold selection degenerates, you must opt in through `threshold_search.allow_degenerate_fallback: true`.

## Probability Calibration

Classifier baselines support:

- `none`
- `sigmoid`
- `isotonic`

Calibration is fit in a time-series-safe way using chronological inner folds on the `train` split.

Why calibration matters here:

- profitable post-cost labels are sparse
- raw classifier scores can be poorly calibrated
- later signal ranking and thresholding are more trustworthy when probabilities are better behaved

The pipeline also writes calibration tables for validation/test when probabilities are available.

## Missing-Data Handling

The old one-size-fits-all median imputation approach has been upgraded.

The pipeline now supports:

- remaining-value median imputation
- optional leakage-safe chronological forward-fill for designated persistent features
- missing-indicator feature generation

Important implementation detail:

- forward-fill is only applied left-to-right in time
- forward-fill is restricted to explicitly designated columns or prefixes
- remaining imputation is still fit on the model training data

This keeps the pipeline practical without introducing backward-looking leakage.

## Prediction Modes

Two broad prediction styles are supported:

### `static`

- fit once on `train`
- score the full dataset with the trained model

This is the simplest and fastest baseline mode.

### `expanding` / `rolling`

- periodically refit in chronological order
- use `prediction.refit_every_n_periods`
- optionally restrict history with `prediction.rolling_window_size`
- optionally exclude validation history from test-period refits

This is still lighter than a full backtest, but it gives more realistic chronological prediction behavior.

## Diagnostics

The baseline module now produces several diagnostics.

### Preferred final interpretation

- held-out permutation importance on validation, or test if validation is unavailable

### Supplemental diagnostics

- linear coefficients for linear models
- impurity-based importances for tree models
- calibration tables for classifier models
- cross-validation search tables
- threshold-search tables

For tree models especially, the report should emphasize permutation importance over impurity importance.

## Strategy-Oriented Evaluation Outputs

The evaluation tables keep the usual ML metrics, but they now emphasize trading usefulness too.

Classification outputs include:

- accuracy
- precision
- recall
- F1
- ROC-AUC
- average precision
- Brier score when probabilities exist

Regression outputs include:

- MAE
- RMSE
- R-squared
- Pearson correlation
- directional accuracy

Both task types now report trading-style metrics such as:

- `signal_count`
- `signal_rate`
- `avg_signal_return_bps`
- `median_signal_return_bps`
- `cumulative_signal_return_bps`
- `signal_hit_rate`
- `precision_among_signaled`
- `signal_sharpe_like`
- `top_quantile_avg_return_bps`

These metrics are usually more meaningful for this project than raw ML accuracy alone.

## Outputs

Default artifact directory:

`data/artifacts/models/baselines/binance/btcusdt/1h/btcusdt_24h_default/`

Key outputs:

- `baseline_predictions.parquet`
  Unified row-level prediction table for rules, linear models, and optional tree baselines.
- `baseline_metrics.parquet`
  Split-level evaluation metrics.
- `baseline_leaderboard.parquet`
  Validation/test comparison summary.
- `baseline_report.md`
  Short markdown report.
- `feature_columns.json`
  Exact final model feature set, including any missing indicators.
- `models/*.joblib`
  Saved model bundles.
- `diagnostics/*`
  Cross-validation results, threshold search tables, permutation importance, coefficients, impurity importance, and calibration tables.
- `baseline_manifest.json`
  Reproducibility metadata including tuned hyperparameters, selected thresholds, calibration choices, prediction mode, and artifact paths.

The report and manifest now also distinguish a healthy model-selection path from a degenerate one, so a strategy row with `signal_count = 0` is accompanied by a machine-readable reason.

## Prediction Table Contract

Downstream signal generation still relies on these columns:

- `timestamp`
- `split`
- `model_name`
- `model_family`
- `task`
- `signal_direction`
- `signal`
- `decision_score`
- `signal_threshold`
- `signal_strength`
- `predicted_probability`
- `predicted_return_bps`
- `predicted_label`
- `actual_label`
- `actual_return_bps`

Additional metadata columns are now included for auditability, such as:

- `selected_hyperparameters_json`
- `selected_threshold_objective`
- `calibration_method`
- `feature_importance_method`
- `prediction_mode`

Those fields are no longer "baseline-only" metadata. They are now propagated into the standardized signal layer and then into backtest / robustness summaries, so later strategy comparisons can explain not just which baseline won, but which configuration decisions produced that result.

## Recommended Review Story

When presenting the baseline layer, a clean narrative is:

1. start with rule-based heuristics
2. show that fixed thresholds are not enough, so validation-driven thresholding matters
3. show that time-series-safe linear models are stronger benchmarks than naive one-shot models
4. show that calibration and held-out permutation importance make the predictions more trustworthy and interpretable
5. then compare these baselines against the later LSTM model

## Caveats

- This is still a prototype benchmark layer, not a live alpha engine.
- Walk-forward prediction here is lighter than the full execution logic in the backtest engine.
- Threshold search is validation-driven, so validation remains a model-selection surface rather than a purely final reporting surface.
- Extremely sparse post-cost positives can still make classifier metrics unstable.
- That sparsity is now surfaced directly through degenerate-experiment diagnostics rather than being hidden inside ordinary-looking zero metrics.
- Tree models remain optional because they are slower and easier to overfit than the penalized-linear baselines.
