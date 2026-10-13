# Data Role Mapping

This document maps the existing `Final Project/data/` files to their expected roles. It is based on Stage 1 filesystem inspection plus the bundled `data/instruction.txt`. Detailed object schemas must be verified in Stage 2 after installing the Python data dependencies.

## Stage 1 Data Tree Summary

Existing project-level files preserved:

- `Guideline.md`
- `Proposal.md`
- `data/`

Existing data folders:

- `data/features/`: 218 `.pkl` characteristic files, approximately 2.99 GB total.
- `data/features500/`: 218 `.pkl` characteristic files, approximately 2.00 GB total.

Stage 1 verification for `features500/`:

- The folder exists under `data/`.
- It has the same number of `.pkl` files as `features/`.
- The first sorted filenames match the corresponding files in `features/`.
- The total byte size is smaller than `features/`.
- `data/instruction.txt` states that `features500` contains comparable data already filtered using the CSI 500 filter at each month-end.

Because the active `python` executable does not currently have `pandas` installed, Stage 1 did not load the pickle or HDF5 objects. Stage 2 should perform schema inspection before panel construction.

## File Mapping

| Data item | Observed in tree | Expected role | Stage 2 inspection needed |
| --- | --- | --- | --- |
| `monthly_returns.pkl` | File present, 15,341,951 bytes | Starting source for monthly stock return targets. Planned target is one-month-ahead excess return after risk-free alignment. | Confirm object type, date index, stock identifier axis, return units, missing values, and whether returns are simple or log returns. |
| `risk_free.csv` | File present, 157,257 bytes. Header observed as `date,RF`. | Risk-free series for excess-return construction. | Confirm frequency, units, compounding, whether daily values must be converted to monthly, and how dates align to month-end returns. |
| `mcap.pkl` | File present, 16,116,402 bytes | Monthly market capitalization panel for size controls, value-weighting, filters, and portfolio diagnostics. | Confirm index and column structure, units, and whether it is point-in-time at month-end. |
| `price.h5` | File present, 324,838,276 bytes | Daily adjusted close prices for price-derived features, return checks, and possibly graph construction diagnostics. | Inspect HDF5 keys, table schema, adjustment convention, and date/asset axes. |
| `BLACKLIST.pkl` | File present, 8,779,858 bytes | Candidate exclusion list by date and asset. `instruction.txt` describes it as a two-level `date` and `asset` index with no additional columns. | Confirm object type, index levels, exact exclusion meaning, and whether dates are daily or monthly. |
| `UNTRADABLE.pkl` | File present, 2,590,987 bytes | Candidate tradability filter by date and asset. `instruction.txt` describes it as a two-level `date` and `asset` index with no additional columns. | Confirm object type, index levels, exact trading restriction meaning, and month-end alignment rule. |
| `csi500_mask_monthly.pkl` | File present, 384,037 bytes | Monthly CSI 500 membership mask for the main universe and for checking `features500/`. `instruction.txt` describes rows as month-end dates, columns as stock tickers, and Boolean values as membership flags. | Confirm date range, column universe, Boolean encoding, and whether it exactly matches non-null coverage in `features500/`. |
| `FF5.csv` | File present, 26,069 bytes. Header observed as `date,MKT,SMB,HML,RMW,CMA`. | Fama-French five-factor benchmark return series for pricing comparison. | Confirm units, date range, month-end alignment, and whether `MKT` is excess market return. |
| `HXZ.csv` | File present, 15,164 bytes. Header observed as `date,MKT,SIZE,INV,ROE`. | Hou-Xue-Zhang factor benchmark return series for pricing comparison. | Confirm units, date range, factor definitions, and month-end alignment. |
| `features500/` | Directory present with 218 `.pkl` files | Default main-spec firm characteristics. Evidence suggests this is a CSI 500-filtered version of `features/`. | Confirm each file's object type, date/asset axes, missingness pattern, feature timing, and whether values outside CSI 500 are absent or masked. |
| `features/` | Directory present with 218 `.pkl` files | Broader-universe firm characteristics for robustness extensions. | Confirm coverage relative to returns and whether the broader universe excludes Beijing exchange as suggested by `mcap.pkl` notes in `instruction.txt`. |

## Additional Observed Data

The data folder also contains `daily_amount.pkl`, `daily_turnover.pkl`, `daily_volume.pkl`, `instruction.txt`, and `zz500.pk`. They are not part of the required Stage 1 mapping list, but they may be useful later for liquidity filters, turnover diagnostics, or CSI 500 cross-checks.
