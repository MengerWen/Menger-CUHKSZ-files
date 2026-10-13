# Robustness Analysis Workflow

## Purpose

The robustness module tests whether the project's strategy conclusions are fragile to reasonable implementation choices and market-friction assumptions.

The current implementation is intentionally practical:

- reuse standardized signals where possible
- reuse the same backtest engine used by the main evaluation flow
- retrain models only when feature ablation actually changes the feature space
- focus on the test split by default so the report reflects out-of-sample behavior

## What The Module Tests

### 1. Cost sensitivity

The report reruns the same strategy families under multiple fee/slippage/gas scenarios.

Default scenarios:

- `low_cost`
- `base_cost`
- `stressed_cost`

This answers:

- does the strategy survive slightly worse execution?
- is the edge mostly a transaction-cost artifact?

### 2. Holding-window sensitivity

The report reruns the same signals with different:

- `holding_window_hours`
- `maximum_holding_hours`

This checks whether conclusions depend too heavily on one chosen horizon such as `24h`.

### 3. Rule-threshold sensitivity

For the rule-based family, the module varies `min_signal_score` at the standardized signal layer.

Important interpretation:

- this is a confidence-threshold robustness test
- it is not a full re-optimization of the original raw rule thresholds

That design keeps the experiment consistent with the unified signal interface and avoids mixing robustness testing with manual strategy redesign.

### 4. Feature ablation

Feature ablation uses the feature-group catalog saved in the feature manifest and removes entire groups such as:

- `funding`
- `basis`
- `volatility`
- `liquidity`
- `interaction_state`

Then it:

1. retrains the simple ML baseline or deep-learning model
2. regenerates signals
3. reruns the same backtest logic

Because feature ablation can easily make an already sparse post-cost task even sparser, the default robustness config now explicitly enables degenerate-fallback handling for ablation retraining. That keeps the report runnable while still surfacing `status`, `reason`, and fallback metadata inside the regenerated artifacts.

Rule-based strategies are not included in feature ablation because they do not directly consume the supervised feature matrix.

## Strategy Families Compared

The default robustness config compares three families:

- `rule_based`: threshold heuristics such as funding and spread rules
- `baseline_ml`: simple predictive models such as logistic/ridge baselines
- `deep_learning`: the first LSTM sequence model

The family comparison section uses the same ranking metric as the report config, which defaults to `cumulative_return`.
The default config now regenerates `rules` and `baseline-ml` signals before running the sweeps so robustness tables stay aligned with the latest upgraded baseline artifacts.

After the baseline upgrade, the robustness tables also preserve the strategy configuration details that matter for interpretation:

- `source_subtype`
  Distinguishes `rule_based`, `baseline_linear`, `baseline_tree`, and `deep_learning`.
- `prediction_mode`
  Shows whether the predictive baseline used static scoring or a more chronological expanding/rolling path.
- `calibration_method`
  Shows whether classifier probabilities were left uncalibrated or calibrated.
- `signal_threshold`
  Shows the threshold that actually converted scores into traded signals.
- `threshold_objective`
  Shows which validation objective selected that threshold.
- `checkpoint_selection_effective_metric`
  Shows how the saved checkpoint was actually chosen after any fallback logic.
- `selected_loss`
  Shows the penalized linear or deep-learning loss that produced the saved model.
- `preprocessing_scaler`
  Shows whether the upstream predictive pipeline used standard or robust scaling.
- `status` / `diagnostic_reason`
  Distinguish a true evaluated strategy from a no-trade row caused by missing tradable signals.

## File / Module Ownership

Recommended ownership for later changes:

- [src/funding_arb/reporting/robustness.py](../src/funding_arb/reporting/robustness.py)
  Main experiment orchestration, report tables, plots, and markdown summary.
- [configs/reports/robustness.yaml](../configs/reports/robustness.yaml)
  Default scenario definitions and output choices.
- [src/funding_arb/cli.py](../src/funding_arb/cli.py)
  CLI wiring for `robustness-report`.
- [src/funding_arb/backtest/engine.py](../src/funding_arb/backtest/engine.py)
  Shared execution and PnL logic reused by robustness sweeps.
- [src/funding_arb/models/baselines.py](../src/funding_arb/models/baselines.py)
  Retraining path for simple-ML feature ablations.
- [src/funding_arb/models/deep_learning.py](../src/funding_arb/models/deep_learning.py)
  Retraining path for deep-learning feature ablations.

## Default Outputs

Presentation-oriented report outputs go under:

- `reports/robustness/<provider>/<symbol>/<frequency>/`

The current command writes:

- `tables/family_comparison_detail.csv`
- `tables/family_comparison_best.csv`
- `tables/cost_sensitivity_detail.csv`
- `tables/cost_sensitivity_best.csv`
- `tables/holding_window_sensitivity_detail.csv`
- `tables/holding_window_sensitivity_best.csv`
- `tables/rule_threshold_sensitivity_detail.csv`
- `tables/feature_ablation_detail.csv`
- `tables/feature_ablation_best.csv`
- `figures/family_comparison.png`
- `figures/cost_sensitivity.png`
- `figures/holding_window_sensitivity.png`
- `figures/rule_threshold_sensitivity.png`
- `figures/feature_ablation.png`
- `summary.json`
- `report.md`

Intermediate model and backtest reruns for robustness live under `data/artifacts/robustness/`.

## How To Run

Run the unified CLI:

```powershell
& 'd:\MG\anaconda3\python.exe' -m src.main robustness-report --config configs/reports/robustness.yaml
```

Compatibility wrapper:

```powershell
& 'd:\MG\anaconda3\python.exe' scripts\reports\robustness_report.py --config configs/reports/robustness.yaml
```

## Assumptions And Limits

- The default report uses the test split only.
- Cost and holding sweeps reuse existing signals; they do not retrain models.
- Threshold sensitivity operates on standardized signal strength rather than raw rule parameter search.
- Feature ablation can be slower because it retrains models.
- Family-level comparisons intentionally keep the high-level family names, but the detail tables now preserve the underlying baseline configuration so the report can explain why one baseline won.
- A no-trade row should now be interpreted through its status fields first, not only through zero-valued numeric metrics.
- The backtest is still the project prototype backtest:
  - single asset
  - delta-neutral abstraction
  - mark-to-market primary equity with realized-only audit columns
  - simplified execution, stop, funding, margin, and order-book assumptions

## Suggested Presentation Framing

When presenting results, a good structure is:

1. show the base family comparison
2. show how performance changes under higher costs
3. show whether the preferred holding window is stable
4. show which feature groups matter most
5. conclude whether the edge appears robust or fragile

That framing keeps the project grounded in realistic research methodology rather than headline backtest numbers alone.
