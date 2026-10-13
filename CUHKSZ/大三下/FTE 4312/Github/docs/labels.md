# Label Generation Specification

## Purpose

This module creates supervised learning targets for the perpetual funding-rate arbitrage project.

The targets are not generic price-direction labels. They are designed to answer a strategy question:

> If we observe features at timestamp `t`, does entering a delta-neutral trade after that signal produce positive post-cost return over a future holding window?

## Core Trade Assumption

The default label direction is:

- `short perp + long spot`

This matches the first prototype strategy focus: capture positive funding and basis convergence while hedging directional exposure.

The code also supports `long_perp_short_spot`, but that side should only be used once borrow assumptions are modeled more carefully.

## Time Alignment

The implementation is explicit about label timing.

For a label at timestamp `t`:

- features are assumed known at the end of bar `t`
- execution is delayed by `execution_delay_bars`, default `1`
- execution prices use the configured field, default `open`
- for a holding window `H`, entry uses bar `t + delay`
- exit uses bar `t + delay + H`

With the current default settings:

- signal timestamp: `t`
- entry price: next bar open, `t + 1`
- exit price for an `8h` label: open at `t + 9`

This is intentionally conservative and avoids same-bar leakage.

## Regression Targets

For each configured horizon `H`, the pipeline produces:

- `target_future_perp_leg_return_bps_{H}h`
- `target_future_spot_leg_return_bps_{H}h`
- `target_future_funding_return_bps_{H}h`
- `target_future_gross_return_bps_{H}h`
- `target_estimated_cost_bps_{H}h`
- `target_future_net_return_bps_{H}h`

### Net-return definition

For the default `short perp + long spot` direction:

```text
future_net_return_bps
= future_perp_leg_return_bps
+ future_spot_leg_return_bps
+ future_funding_return_bps
- estimated_cost_bps
```

This is an opportunity-quality target in basis points of per-leg notional.

## Classification Targets

For each horizon `H`, the pipeline also produces:

- `target_is_profitable_{H}h`
- `target_is_tradeable_{H}h`

Definitions:

- `target_is_profitable_{H}h = 1` if `target_future_net_return_bps_{H}h > positive_return_threshold_bps`
- `target_is_tradeable_{H}h = 1` if `target_future_net_return_bps_{H}h > min_expected_edge_bps`

With the default config:

- profitable threshold: `0.0` bps
- tradeable threshold: `5.0` bps

So the first classification target answers whether the trade is post-cost profitable at all, while the second answers whether it clears a more practical execution edge threshold.

## Cost Model in the Labels

The first label version includes these estimated frictions:

- taker fees
- slippage
- gas converted into basis points using `position_notional_usd`
- optional flat friction term via `other_friction_bps`
- optional borrow-cost term via `borrow_cost_bps_per_hour`

Current trading-cost approximation:

```text
estimated_cost_bps
= 4 * (taker_fee_bps + slippage_bps)
+ gas_cost_bps
+ other_friction_bps
+ borrow_cost_bps_if_applicable
```

Why the factor `4`:

- perp entry
- perp exit
- hedge entry
- hedge exit

This is simple, explicit, and easy to adjust later.

## Funding Accrual Approximation

Funding is summed over the next `H` labeled hourly rows after the execution delay.

This is a bar-based approximation rather than exact intrahour execution accounting. It is acceptable for the first research pipeline, but should be refined later if the project needs more exact funding timestamp treatment.

## Time-Series Split Logic

The supervised dataset uses chronological splits only.

Default split boundaries:

- train end: `2024-06-30`
- validation end: `2025-03-31`
- test end: `2026-04-07`

Split assignment is inclusive by boundary:

- `timestamp <= train_end` -> `train`
- `train_end < timestamp <= validation_end` -> `validation`
- `validation_end < timestamp <= test_end` -> `test`
- later timestamps -> `excluded`

## Saved Outputs

Default outputs go under:

- `data/processed/supervised/binance/btcusdt/1h/`

Artifacts:

- combined supervised dataset: `btcusdt_supervised_dataset.parquet`
- label-only table: `btcusdt_label_table.parquet`
- manifest: `btcusdt_supervised_manifest.json`
- split datasets: `splits/train.parquet`, `splits/validation.parquet`, `splits/test.parquet`

The manifest now also records split-level diagnostics that matter for downstream model selection:

- `label_diagnostics_by_split`
- `tradeable_rate_by_split`
- `profitable_rate_by_split`
- `degenerate_experiment`
- `degenerate_stage`
- `degenerate_reason`

The combined supervised dataset contains:

- feature columns from the feature pipeline
- label columns for each configured horizon
- `split`
- `supervised_ready`

`supervised_ready = 1` only when:

- the row is feature-ready
- the primary regression target is available
- the row belongs to train, validation, or test

## Config Entry Point

Main config:

- `configs/labels/default.yaml`

Main command:

```bash
& 'd:\MG\anaconda3\python.exe' -m src.main build-labels --config configs/labels/default.yaml
```

## Practical Notes

- Final rows near the end of the sample will naturally have `NaN` labels for longer horizons.
- This is expected and is handled through `supervised_ready` rather than by silently dropping rows from the combined dataset.
- The first label version is deliberately interpretable and cost-aware. It is a strong prototype target for both rule baselines and deep learning models.

## Degenerate Split Detection

The label pipeline now explicitly diagnoses splits that cannot support sensible downstream threshold selection.

Examples:

- `tradeable_rate == 0`
- `profitable_rate == 0`
- the relevant `future_net_return_bps` values never exceed the configured tradable edge threshold

These cases no longer stay implicit. The manifest flags them so later training stages can either:

- fail fast when validation cannot support threshold selection, or
- continue only with an explicit warning or opt-in fallback setting

This design keeps the repository honest about sparse post-cost labels instead of letting later reports hide the issue behind normal-looking zeroes.
