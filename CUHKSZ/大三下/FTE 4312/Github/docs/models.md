# Deep Learning Models

## Purpose

This module is the sequence-modeling research layer for the project. It is designed to answer a practical question:

Can a small time-series model learn better post-cost opportunity signals from the engineered hourly feature set than simple rule-based and linear baselines?

The implementation deliberately stays small, but it is no longer locked to one exact architecture.

Phase 1 adds a compact model zoo while preserving:

- the current `train-dl` CLI workflow
- the current prediction artifact contract
- the current signal/backtest integration path
- the current time-series-safe dataset and normalization logic

Phase 2 adds a compact comparison layer on top of those per-model artifacts:

- one YAML-driven `compare-dl` workflow
- named experiment bundles for regression, classification, and recurrent-only comparisons
- aggregated validation/test/strategy leaderboards
- comparison markdown, CSV/parquet tables, plots, and manifests

The current default is still `LSTM`. The comparison layer deliberately reuses the per-model artifact format rather than introducing a separate experiment-tracking platform.

## Default Task Setup

The default config uses regression rather than classification:

- task: `regression`
- target column: `target_future_net_return_bps_24h`
- reference classification column: `target_is_profitable_24h`
- lookback window: `48` hourly steps

Why regression first:

- the 24-hour net-return target is less sparse than the positive classification labels
- it gives a ranked expected-return output that is easy to feed into later backtests
- it still stays aligned with tradable post-cost opportunity quality

The same pipeline also supports classification by changing `target.task` and `target.column` in `configs/models/lstm.yaml`.

## Architecture

Implemented sequence models:

- `LSTMSequenceModel`
- `GRUSequenceModel`
- `TCNSequenceModel`
- `TransformerEncoderSequenceModel`

Shared design:

- one shared `SequenceDataset`
- one shared training loop
- one shared prediction output schema
- one shared artifact/report/manifest structure
- one dispatch-based model builder selected by `model.name`

Family-specific summary:

- `lstm`
  Stacked recurrent encoder with final hidden state and scalar output head.
- `gru`
  Same overall interface as LSTM, but with a lighter recurrent cell.
- `tcn`
  Causal dilated Conv1d stack with compact residual blocks and final-step readout.
- `transformer_encoder`
  Input projection + sinusoidal positional encoding + causal encoder stack + scalar head.

Default LSTM reference hyperparameters:

- hidden size: `64`
- layers: `2`
- dropout: `0.1`
- bidirectional: `false`

The model builder is dispatch-based, so model-family expansion does not require changes to sequence construction, artifact format, or CLI usage.

## Practical Trade-Offs

### LSTM vs GRU

- `LSTM`
  More expressive gating and the most established reference point in this repository.
- `GRU`
  Fewer parameters and often easier to train, which can be attractive when the edge is weak and the dataset is not huge.

### Recurrent vs TCN

- recurrent models summarize history step by step and are a natural fit for ordered funding/basis regimes
- `TCN` is more parallelizable and often trains stably, but it relies on a finite convolutional receptive field rather than recurrent memory

### Transformer under limited financial data

- `transformer_encoder` can model flexible long-range interactions inside the lookback window
- but it is also the easiest family here to overfit or become unnecessarily heavy on a modest hourly research dataset
- for this course project, it should be treated as a compact benchmark, not as a claim that “bigger attention models are automatically better”

## Training and Selection Logic

The first prototype selected checkpoints only by validation loss. The upgraded version is more research-oriented.

It now supports configurable checkpoint selection metrics such as:

- `validation_loss`
- `validation_f1`
- `validation_roc_auc`
- `validation_pearson_corr`
- `validation_avg_signal_return_bps`
- `validation_signal_hit_rate`
- `validation_cumulative_signal_return_bps`

For the default regression experiment, the config now uses a trading-aware selection metric so the saved checkpoint is chosen with signal usefulness in mind rather than loss alone.

If a trading-oriented validation metric becomes undefined because an epoch produces zero traded signals, the pipeline still computes the fallback-to-loss diagnostic internally so the history remains auditable. However, the run no longer silently accepts that fallback by default. If the final selected checkpoint would only be chosen via fallback, the training command now fails unless `training.allow_degenerate_fallback: true` is set explicitly.

Thresholds are also no longer fixed-only. The module can search thresholds on the validation split:

- probability thresholds for classification
- predicted-return thresholds for regression

The default threshold objective is `avg_signal_return_bps`.

Threshold search also now fails fast by default when:

- validation label diagnostics show no support for threshold selection
- every candidate threshold has an invalid objective
- the only way to continue would be a silent default-threshold fallback

To continue anyway for diagnostic artifact generation, you must opt in through `threshold_search.allow_degenerate_fallback: true`.

## Input Shape

The model consumes a 3D tensor with shape:

`[batch_size, lookback_steps, feature_count]`

Under the current default config:

- `lookback_steps = 48`
- `feature_count` is determined from the supervised dataset after leakage-safe feature selection

Each target row uses:

- the feature row at time `t`
- the preceding `lookback_steps - 1` rows
- no future rows

That means the sequence ending at time `t` predicts the label already aligned by the supervised dataset pipeline, so there is no future leakage from sequence construction.

## Time-Series Safety

The deep-learning data path is intentionally conservative:

- rows are sorted strictly by timestamp before sequence construction
- only rows with `supervised_ready == 1` are eligible as targets
- feature normalization is fit on the training split only
- missing and infinite values are handled before tensor conversion
- validation and test sequences use earlier historical rows only
- no future labels or future features are exposed to the model
- TCN uses left-padding-only causal convolutions
- TransformerEncoder uses an explicit causal attention mask

By default, `allow_cross_split_context = true`. This means a validation or test target may use preceding historical rows from earlier splits as context. This is intentional and realistic: when we score a later time, we do know the earlier market history.

## Outputs

Default output directory:

`data/artifacts/models/dl/binance/btcusdt/1h/lstm_regression_24h_default/`

Key artifacts:

- `best_model.pt`
  Best checkpoint selected by the configured validation metric.
- `training_history.csv`
  Per-epoch train/validation loss, selected threshold, selection score, and strategy-oriented metrics.
- `dl_predictions.parquet`
  Row-level predictions and trading signals.
- `dl_metrics.parquet`
  Split-level evaluation summary.
- `dl_leaderboard.parquet`
  Validation/test leaderboard view.
- `feature_columns.json`
  Exact input feature list.
- `feature_normalization.json`
  Training-only normalization statistics.
- `training_report.md`
  Lightweight experiment report.
- `dl_manifest.json`
  Reproducibility summary including selected loss, threshold, configured checkpoint metric, effective checkpoint metric, prediction mode, preprocessing, and tuning settings.

Additional diagnostics may also be written under `diagnostics/`, including:

- threshold-search tables
- tuning results when enabled
- classification calibration tables
- feature-group ablation summaries

The training report and manifest now also expose:

- `degenerate_experiment`
- `degenerate_stage`
- `degenerate_reason`
- `fallback_used`
- `fallback_reason`
- `signal_count_by_split`
- `tradeable_rate_by_split`
- `profitable_rate_by_split`
- `threshold_search_summary`

## Prediction Output Contract

The prediction table follows the same general format as the baseline pipeline so later backtests can consume both:

- `timestamp`
- `split`
- `model_name`
- `model_family`
- `task`
- `signal`
- `decision_score`
- `signal_threshold`
- `signal_strength`
- `predicted_probability`
- `predicted_return_bps`
- `actual_label`
- `actual_return_bps`

Additional metadata columns now include:

- `selected_hyperparameters_json`
- `selected_threshold_objective`
- `selected_threshold_objective_value`
- `calibration_method`
- `feature_importance_method`
- `prediction_mode`
- `checkpoint_selection_metric`
- `checkpoint_selection_effective_metric`
- `checkpoint_selection_fallback_used`
- `selected_loss`
- `preprocessing_scaler`

These fields are carried forward so the signal layer, backtest engine, and robustness report can explain not only model performance, but also how the saved deep-learning checkpoint was selected and preprocessed.

For regression:

- `decision_score` and `predicted_return_bps` are the same
- `signal = 1` when predicted return exceeds `trade_threshold_bps`

For classification:

- `decision_score` is the predicted probability
- `signal = 1` when probability exceeds `probability_threshold`

## CLI

Single-model training through the existing unified CLI:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main train-dl --config configs/models/lstm.yaml
```

Other model-family configs:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main train-dl --config configs/models/gru.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main train-dl --config configs/models/tcn.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main train-dl --config configs/models/transformer.yaml
```

Wrapper script:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts\models\train_dl.py --config configs/models/lstm.yaml
```

Multi-model comparison through the Phase 2 workflow:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main compare-dl --config configs/experiments/dl/regression_all.yaml
```

Other named bundles:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main compare-dl --config configs/experiments/dl/recurrent_regression.yaml
& 'd:\MG\anaconda3\python.exe' -m src.main compare-dl --config configs/experiments/dl/classification_all.yaml
```

Comparison wrapper script:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts\models\compare_dl.py --config configs/experiments/dl/regression_all.yaml
```

Important single-model configs:

- [lstm.yaml](../configs/models/lstm.yaml)
  Default reference recurrent experiment.
- [gru.yaml](../configs/models/gru.yaml)
  Lighter recurrent benchmark.
- [tcn.yaml](../configs/models/tcn.yaml)
  Compact convolutional sequence benchmark.
- [transformer.yaml](../configs/models/transformer.yaml)
  Compact causal attention benchmark.

Important comparison bundle configs:

- [regression_all.yaml](../configs/experiments/dl/regression_all.yaml)
  Compares `lstm`, `gru`, `tcn`, and `transformer_encoder` on the 24-hour post-cost net-return regression target.
- [classification_all.yaml](../configs/experiments/dl/classification_all.yaml)
  Reuses the same four model configs with explicit classification overrides for `target_is_profitable_24h`.
- [recurrent_regression.yaml](../configs/experiments/dl/recurrent_regression.yaml)
  Compares only the recurrent reference models, `lstm` and `gru`.

## Phase 2 Comparison Outputs

Default output directory:

`data/artifacts/models/dl_comparisons/binance/btcusdt/1h/sequence_regression_all/`

Key artifacts:

- `comparison_summary.parquet` / `comparison_summary.csv`
  One row per model run with task, target, lookback, selected hyperparameters, best epoch, checkpoint metric, selected threshold, validation/test metrics, and artifact paths.
- `validation_leaderboard.parquet` / `validation_leaderboard.csv`
  Ranking table using the bundle's configured validation metric.
- `test_leaderboard.parquet` / `test_leaderboard.csv`
  Ranking table using the bundle's configured test metric.
- `strategy_leaderboard.parquet` / `strategy_leaderboard.csv`
  Ranking table using a strategy-oriented metric such as top-quantile or signal-return quality.
- `comparison_report.md`
  Human-readable report listing models, configs, headline results, trade-offs, and the current default best model under the configured test metric.
- `comparison_manifest.json`
  Reproducibility manifest with bundle settings, output paths, and per-model artifact locations.
- `figures/`
  Compact bar charts for validation, test, and strategy-oriented comparison metrics.

The comparison runner first checks whether each per-model artifact already exists. If it does, the runner reuses that artifact by default. If it is missing and `runner.train_if_missing=true`, it trains that model through the existing `train-dl` pipeline. Set `runner.force_retrain_all=true` or per-run `force_retrain=true` when you intentionally want to refresh model artifacts.

The unified signal config currently points its default `dl` source at the current Phase 2 regression comparison winner. The single-model `train-dl` default remains LSTM so the project still has a stable reference model, while downstream `generate-signals --source dl` can consume the best available comparison artifact.

## How It Differs From Baselines

Baseline models treat each timestamp mostly as an independent row after feature engineering.
The sequence models instead learn from ordered windows of feature history.

In practice, that means the sequence-model family can model:

- short-term temporal persistence in funding and basis behavior
- multi-hour build-up and unwind patterns
- interactions that depend on sequence order rather than only the current row snapshot

At the same time, the current implementation remains lightweight:

- one compact model zoo
- one target at a time per run
- no distributed training
- no experiment server
- no complex scheduler stack

That keeps the module maintainable and appropriate for a course-project prototype.

## Walk-Forward and Robustness Hooks

The upgraded module also adds two realism/robustness hooks:

- optional tuning-ready time-series validation structure
- optional expanding / rolling prediction mode with periodic retraining

The default run still uses `prediction.mode = static` to keep the main experiment easy to reproduce, but the code is now prepared for more chronological scoring when needed.

Preprocessing can also be made more robust through:

- winsorization before scaling
- `standard` or `robust` scaling
- training-only fitted preprocessing statistics

## Interpretability

This is still not a SHAP-heavy deep-learning research stack, but the module now includes a practical interpretability path:

- feature-group ablation on validation/test predictions

Groups are taken from the feature manifest when available, and otherwise inferred heuristically from feature names.

## Caveats

- The current post-cost labels are intentionally hard, so model quality should be judged with both predictive metrics and realized signal-return diagnostics.
- This first version uses only engineered tabular features arranged into sequences; it does not yet include raw order-book or multi-asset sequence inputs.
- Tuning support is intentionally lightweight and disabled by default because full deep time-series hyperparameter search can become expensive quickly.
- Walk-forward mode is available, but the exported checkpoint still refers to the base train/validation model rather than every refit chunk.
- Phase 2 adds multi-model comparison workflow and reporting, but it is still a compact course-project suite, not a full experiment-tracking server.
- Artifact reuse is intentionally lightweight. If a config override changes but the same `output.run_name` already exists, set `force_retrain=true` to avoid unintentionally comparing stale artifacts.
