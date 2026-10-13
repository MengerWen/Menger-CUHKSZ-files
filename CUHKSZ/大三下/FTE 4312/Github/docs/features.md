# Feature Engineering Specification

## Purpose

This document defines the first practical and interpretable feature set for the perpetual funding-rate arbitrage research pipeline.

The current implementation reads the canonical hourly dataset produced by `fetch-data` and writes a feature table under `data/processed/features/<provider>/<symbol>/<frequency>/`.

## Leakage Control Assumption

All features are computed using information available at or before timestamp `t` only.

- rolling statistics use current and past observations only
- no feature uses future funding prints, future returns, or future spread values
- the intended downstream modeling assumption is: feature values at timestamp `t` are used to make a decision for execution after that bar closes, typically on the next bar

## Output Columns

The final feature table preserves key base columns from the canonical dataset and appends engineered features.

### Base columns preserved

- `timestamp`
- `symbol`
- `venue`
- `frequency`
- `perp_close`
- `spot_close`
- `funding_rate`
- `funding_event`
- `perp_volume`
- `spot_volume`
- `open_interest`
- `perp_close_was_missing`
- `spot_close_was_missing`
- `open_interest_was_missing`
- `feature_ready`

`feature_ready` becomes `1` only after the core rolling features required by the current config are available.

## Feature Groups

The default config uses rolling windows `{8, 24, 72, 168}` hours, with primary windows:

- volatility window: `24h`
- z-score window: `72h`
- funding mean window: `24h`
- basis mean window: `24h`
- shock window: `24h`
- liquidity window: `24h`
- regime window: `168h`

### 1. Funding-related features

| Feature columns | Definition | Interpretation |
| --- | --- | --- |
| `funding_rate_raw` | Raw hourly funding rate from the canonical dataset | Core carry signal |
| `funding_rate_bps` | `funding_rate_raw * 10000` | Easier-to-read funding level in basis points |
| `funding_annualized_proxy` | `funding_rate_raw * (24 / funding_interval_hours) * 365` | Annualized carry proxy under the exchange funding interval assumption |
| `funding_sign` | Sign of `funding_rate_raw` in `{-1, 0, 1}` | Direction of the carry regime |
| `funding_sign_reversal` | `1` when consecutive non-zero funding observations flip sign | Detects funding regime turns |
| `funding_event_flag` | Mirror of the canonical `funding_event` indicator | Distinguishes real funding settlement hours from zero-filled hours |
| `funding_mean_8h`, `funding_mean_24h`, `funding_mean_72h`, `funding_mean_168h` | Rolling mean of `funding_rate_bps` | Persistence of carry conditions |
| `funding_std_8h`, `funding_std_24h`, `funding_std_72h`, `funding_std_168h` | Rolling standard deviation of `funding_rate_bps` | Stability vs abnormality of funding |
| `funding_zscore_8h`, `funding_zscore_24h`, `funding_zscore_72h`, `funding_zscore_168h` | Rolling z-score of `funding_rate_bps` | Relative funding extremeness |
| `funding_positive_share_8h`, `funding_positive_share_24h`, `funding_positive_share_72h`, `funding_positive_share_168h` | Rolling share of positive funding observations | Persistence of long-crowded conditions |

### 2. Basis / spread features

| Feature columns | Definition | Interpretation |
| --- | --- | --- |
| `spread_usd` | `perp_close - spot_close` | Raw perp-vs-spot dislocation |
| `spread_bps` | `((perp_close / spot_close) - 1) * 10000` | Normalized spread in basis points |
| `spread_change_1h` | 1-hour change in `spread_bps` | Short-horizon spread momentum |
| `spread_change_24h` | 24-hour change in `spread_bps` | Medium-horizon spread drift |
| `spread_mean_8h`, `spread_mean_24h`, `spread_mean_72h`, `spread_mean_168h` | Rolling mean of `spread_bps` | Local equilibrium estimate |
| `spread_std_8h`, `spread_std_24h`, `spread_std_72h`, `spread_std_168h` | Rolling standard deviation of `spread_bps` | Spread dispersion |
| `spread_zscore_8h`, `spread_zscore_24h`, `spread_zscore_72h`, `spread_zscore_168h` | Rolling z-score of `spread_bps` | Relative spread abnormality |
| `spread_deviation_8h`, `spread_deviation_24h`, `spread_deviation_72h`, `spread_deviation_168h` | `spread_bps - spread_mean_wh` | Distance from local fair value |
| `spread_reversion_signal_8h`, `spread_reversion_signal_24h`, `spread_reversion_signal_72h`, `spread_reversion_signal_168h` | Negative of `spread_zscore_wh` | Mean-reversion-oriented score |

### 3. Volatility / risk features

| Feature columns | Definition | Interpretation |
| --- | --- | --- |
| `perp_return_1h` | Hourly percent return of `perp_close` | Perp short-horizon price move |
| `spot_return_1h` | Hourly percent return of `spot_close` | Hedge-leg short-horizon price move |
| `perp_abs_return_1h` | Absolute value of `perp_return_1h` | Realized move magnitude |
| `spot_abs_return_1h` | Absolute value of `spot_return_1h` | Realized move magnitude |
| `perp_realized_vol_8h`, `perp_realized_vol_24h`, `perp_realized_vol_72h`, `perp_realized_vol_168h` | Annualized rolling volatility of `perp_return_1h` | Risk regime for the perp leg |
| `spot_realized_vol_8h`, `spot_realized_vol_24h`, `spot_realized_vol_72h`, `spot_realized_vol_168h` | Annualized rolling volatility of `spot_return_1h` | Risk regime for the hedge leg |
| `perp_return_shock_24h` | `abs(perp_return_1h) / rolling_std(perp_return_1h, 24)` | Short-term perp shock intensity |
| `spot_return_shock_24h` | `abs(spot_return_1h) / rolling_std(spot_return_1h, 24)` | Short-term spot shock intensity |

### 4. Liquidity / activity features

| Feature columns | Definition | Interpretation |
| --- | --- | --- |
| `perp_volume_raw` | Raw hourly perp volume | Direct activity proxy |
| `spot_volume_raw` | Raw hourly spot volume | Direct hedge-leg activity proxy |
| `perp_dollar_volume_raw` | `perp_volume * perp_close` | Notional activity in the perp leg |
| `spot_dollar_volume_raw` | `spot_volume * spot_close` | Notional activity in the spot leg |
| `perp_volume_change_1h` | Hourly percent change in perp volume | Activity acceleration |
| `spot_volume_change_1h` | Hourly percent change in spot volume | Activity acceleration |
| `perp_volume_ratio_8h`, `perp_volume_ratio_24h`, `perp_volume_ratio_72h`, `perp_volume_ratio_168h` | `perp_volume / rolling_mean(perp_volume, w)` | Current perp activity relative to recent norm |
| `spot_volume_ratio_8h`, `spot_volume_ratio_24h`, `spot_volume_ratio_72h`, `spot_volume_ratio_168h` | `spot_volume / rolling_mean(spot_volume, w)` | Current spot activity relative to recent norm |
| `open_interest_raw` | Raw open interest from the canonical dataset, if available | Crowding proxy |
| `open_interest_change_1h` | Hourly percent change in open interest, if available | Position build-up or unwind |
| `open_interest_ratio_8h`, `open_interest_ratio_24h`, `open_interest_ratio_72h`, `open_interest_ratio_168h` | `open_interest / rolling_mean(open_interest, w)`, if available | Relative crowding |
| `open_interest_zscore_8h`, `open_interest_zscore_24h`, `open_interest_zscore_72h`, `open_interest_zscore_168h` | Rolling z-score of open interest, if available | Abnormal crowding state |

Open-interest features are only generated when the input dataset contains non-null `open_interest` values.

### 5. Interaction / state features

| Feature columns | Definition | Interpretation |
| --- | --- | --- |
| `funding_x_perp_vol_24h` | `funding_rate_bps * perp_realized_vol_24h` | Carry under risk stress |
| `funding_x_spread_bps` | `funding_rate_bps * spread_bps` | Interaction between carry and basis dislocation |
| `spread_x_perp_vol_24h` | `spread_bps * perp_realized_vol_24h` | Dislocation under elevated risk |
| `positive_funding_regime` | `1` when `funding_mean_24h > 0` | Long-crowded carry regime |
| `high_vol_regime` | `1` when `perp_realized_vol_24h` is above its rolling 168h median | Elevated volatility regime |
| `wide_spread_regime` | `1` when `abs(spread_zscore_72h) > 1` | Material basis dislocation regime |
| `shock_regime` | `1` when `perp_return_shock_24h > 2` | Short-term disturbance regime |
| `funding_x_open_interest` | `funding_rate_bps * open_interest_raw`, if open interest exists | Carry under crowding pressure |

## Output Artifact Paths

Default output paths:

- feature table: `data/processed/features/binance/btcusdt/1h/btcusdt_feature_set.parquet`
- optional CSV mirror: `data/processed/features/binance/btcusdt/1h/btcusdt_feature_set.csv`
- manifest: `data/processed/features/binance/btcusdt/1h/btcusdt_feature_manifest.json`

## Practical Notes

- Early rows will contain `NaN` values for long-window features until enough history accumulates.
- This is expected and is surfaced through `feature_ready` rather than silently dropping those rows.
- The current feature set is deliberately interpretable and course-project friendly; future iterations can add premium-index, mark-price, microstructure, or cross-venue features without changing the pipeline structure.