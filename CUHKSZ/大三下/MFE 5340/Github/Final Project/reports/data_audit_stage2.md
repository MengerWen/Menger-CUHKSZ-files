# Stage 2 Data Audit

## Data Inspection Summary

Stage 2 inspected the actual files under `Final Project/data/` using `d:\MG\anaconda3\python.exe` with pandas/numpy available. The detailed machine-readable inspection output is saved at `outputs/metadata/data_inspection_stage2.json`.

Confirmed raw structures:

| File | Confirmed structure |
| --- | --- |
| `monthly_returns.pkl` | pandas DataFrame, shape `(355, 5378)`, monthly `DatetimeIndex` named `date`, stock-code columns such as `000001.SZ`, span `1995-02-28` to `2024-08-31`. |
| `mcap.pkl` | pandas DataFrame, shape `(356, 5634)`, monthly `DatetimeIndex` named `date`, stock-code columns, span `1995-01-31` to `2024-08-31`. |
| `BLACKLIST.pkl` | pandas DataFrame, shape `(2161751, 0)`, `MultiIndex` with levels `date` and `asset`, daily dates from `1991-12-16` to `2024-09-11`. |
| `UNTRADABLE.pkl` | pandas DataFrame, shape `(621327, 0)`, `MultiIndex` with levels `date` and `asset`, daily dates from `1990-12-20` to `2024-09-11`. |
| `csi500_mask_monthly.pkl` | pandas DataFrame, shape `(214, 1687)`, monthly `DatetimeIndex` named `date`, Boolean stock-code columns, span `2007-01-31` to `2024-10-31`. |
| `risk_free.csv` | CSV with columns `date, RF`, 7,223 rows; dates are daily strings and `RF` is numeric. |
| `FF5.csv` | CSV with columns `date, MKT, SMB, HML, RMW, CMA`, 357 monthly rows. |
| `HXZ.csv` | CSV with columns `date, MKT, SIZE, INV, ROE`, 250 monthly rows. |
| `price.h5` | HDF5 file with key `/price`. It was inspected at the store/key level only in Stage 2 because the monthly panel does not require daily prices yet. |

Representative feature files from both `features500/` and `features/` are monthly wide DataFrames with month-end `DatetimeIndex` and stock-code columns. Examples inspected include `abnormal_turnover_21.pkl`, `total_mv.pkl`, `west_eps_fy1.pkl`, and `return_on_invested_capital_ttm.pkl`.

## Universe Finding

`features500/` is retained as the default main specification. This is now supported by actual-file inspection, not just naming:

- `features500/abnormal_turnover_21.pkl` spans `2007-01-31` to `2024-08-31` with 212 monthly rows.
- The broader `features/abnormal_turnover_21.pkl` spans `1998-01-31` to `2024-08-31` with 320 monthly rows.
- In the full common date/stock comparison between `features500/abnormal_turnover_21.pkl` and `csi500_mask_monthly.pkl`, there were `102,463` non-null feature observations inside the CSI 500 mask and `0` non-null observations outside the mask.
- `features500/` still keeps broad stock-code columns, but values outside CSI 500 membership appear to be stored as missing values.

## Panel Artifact

The cleaned main-spec panel is saved at:

```text
outputs/panels/main_features500_panel.pkl
```

Metadata is saved at:

```text
outputs/metadata/main_features500_panel_metadata.json
```

Final panel summary:

| Item | Value |
| --- | ---: |
| Rows | 90,351 |
| Columns | 227 |
| Months | 211 |
| Date span | `2007-01-31` to `2024-07-31` |
| Unique stocks | 1,678 |
| Stocks per month, min | 220 |
| Stocks per month, median | 448 |
| Stocks per month, max | 499 |
| Stocks per month, mean | 428.20 |
| Raw feature files loaded | 218 |
| Features kept | 218 |
| Features dropped | 0 |

The panel columns include:

- `date`
- `stock_id`
- `csi500_member_t`
- `target_return`
- `rf_next_month`
- `target_excess_return`
- `mcap_t`
- `blacklisted_t`
- `untradable_t`
- 218 cleaned firm characteristics

## Target Construction

For each stock-month row dated month `t`:

- `target_return` is the realized value from `monthly_returns.pkl` at month `t + 1`.
- `rf_next_month` is the compounded daily `RF` from `risk_free.csv` over month `t + 1`.
- `target_excess_return = target_return - rf_next_month`.

The final panel drops rows with missing `target_return` or missing `target_excess_return`. This is why the final panel ends at `2024-07-31`: the last available monthly return is `2024-08-31`, so `2024-07-31` is the last month with a one-month-ahead target.

Final target missingness is zero for both target columns. The final-sample mean and standard deviation are:

| Target | Mean | Std. dev. |
| --- | ---: | ---: |
| `target_return` | 0.011961 | 0.133740 |
| `target_excess_return` | 0.010139 | 0.133796 |

## Cleaning and Filtering

The cleaning choices are configured in `configs/cleaning_features500.yaml`.

Implemented default steps:

- Use `features500/` as the feature source.
- Use the CSI 500 membership mask at month `t` as the base universe.
- Convert daily `BLACKLIST.pkl` records to current-month flags using `any_in_current_month`.
- Convert daily `UNTRADABLE.pkl` records to current-month flags using `any_in_current_month`.
- Filter out rows flagged as blacklisted at month `t`.
- Filter out rows flagged as untradable at month `t`.
- Drop rows without next-month return targets and next-month risk-free rates.
- Drop feature columns only if their post-filter missing fraction exceeds `0.95`; no features were dropped under this rule.
- Winsorize features cross-sectionally by month at the 1st and 99th percentiles.
- Impute feature missing values using the cross-sectional median within each month, falling back to the global feature median and then zero only if needed.
- Normalize features cross-sectionally by month to mean zero and unit standard deviation.

Row counts through the main filters:

| Step | Rows |
| --- | ---: |
| Base CSI 500 stock-month rows | 105,500 |
| After blacklist filter | 104,053 |
| After untradable filter | 90,351 |
| After dropping missing target | 90,351 |
| After dropping missing excess target | 90,351 |
| After feature missing filters | 90,351 |

Base flag counts before filtering:

| Flag | Count |
| --- | ---: |
| `blacklisted_t` | 1,447 |
| `untradable_t` | 14,014 |

After filtering, both final flag columns are retained but are all `False` by construction. `mcap_t` is retained where available and has a final missing fraction of approximately `0.000476`; it is not imputed in Stage 2.

Top raw feature missing fractions before filtering and imputation:

| Feature | Missing fraction |
| --- | ---: |
| `free_cash_flow_yoy_pct_chg` | 0.883611 |
| `rdexp_to_asset_ttm` | 0.654834 |
| `rdexp_to_total_mktcap_ttm` | 0.654834 |
| `west_eps_fy1` | 0.456692 |
| `dispersion_in_analyst_earnings_forecasts` | 0.287754 |
| `west_stdeps_fy1` | 0.287744 |
| `west_netprofit_fy1_1m` | 0.280474 |
| `changes_in_analyst_earnings_forecasts` | 0.270938 |
| `west_eps_fy1_chg_1m` | 0.262123 |
| `rating_avg` | 0.253384 |

After feature imputation and normalization, the final feature matrix has zero missing values.

## Time Alignment and Look-Ahead Control

The panel uses feature values timestamped at month `t` to predict returns realized at `t + 1`. Market cap, CSI 500 membership, blacklist flags, and untradable flags are also taken from month `t`, not from `t + 1`.

Cleaning is performed cross-sectionally within each month:

- Winsorization thresholds are computed using only stocks in that month.
- Imputation medians are computed using only stocks in that month, with a global fallback only for features that are entirely missing in a month.
- Normalization means and standard deviations are computed using only stocks in that month.

This controls the main mechanical sources of look-ahead bias in panel construction. The remaining limitation is feature disclosure timing: the stored feature matrices provide month-end timestamps but do not reveal announcement dates or reporting lags for every characteristic. Stage 2 therefore assumes that a feature value dated month `t` is the value available for a signal formed at month `t`, and explicitly documents this as an unresolved caveat.

## Confirmed Facts vs Assumptions

Confirmed facts from file inspection:

- Main return, market-cap, mask, and feature files are pandas DataFrames with date-based indices and stock-code columns.
- Stock identifiers use strings such as `000001.SZ` and `600000.SH`.
- `BLACKLIST.pkl` and `UNTRADABLE.pkl` are daily zero-column MultiIndex DataFrames with index levels `date` and `asset`.
- `risk_free.csv` is daily and has columns `date` and `RF`.
- `features500/` appears prefiltered to the CSI 500 universe through missing values outside the mask, while retaining broad stock-code columns.

Assumptions used for the Stage 2 panel:

- Monthly feature timestamps represent information available at month `t` for predicting month `t + 1`.
- Daily `RF` values should be compounded to monthly risk-free returns before subtracting from next-month stock returns.
- Daily blacklist and untradable records can be aggregated to current-month flags using any occurrence in month `t`.
- For the main specification, explicit `csi500_mask_monthly.pkl` filtering is used even though `features500/` already appears filtered, because this makes the universe rule transparent.

Unresolved caveats for later stages:

- Verify disclosure lags for accounting and analyst-based features if source metadata becomes available.
- Inspect `price.h5` values more deeply before using daily prices in graph construction or return validation.
- Decide whether rows with missing `mcap_t` should be dropped or imputed for portfolio weighting.
- Consider stricter feature missingness thresholds or feature family selection before final model training.
