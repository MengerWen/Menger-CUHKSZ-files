# Data Source and Processing

## Data Source

The project uses the local `Data.csv` file supplied with the course. The file states that it was created from the 202601 CRSP database and contains value-weighted returns for six portfolios formed on market equity and book-to-market equity.

## Cleaning Choices

- The raw CSV begins with descriptive text, so the cleaning script locates the actual monthly value-weighted header row programmatically instead of relying on a fragile fixed row count.
- The missing-value sentinels documented in the file, `-99.99` and `-999`, are converted to `NaN` during parsing.
- The raw `YYYYMM` date code is converted into a proper month-end `date` column.
- The canonical cleaned dataset stores returns in percent units with `_pct` suffixes. A companion decimal dataset divides those columns by `100.0` and uses `_dec` suffixes.

## Sanity Checks

- Sample coverage: 1926-07-31 to 2026-01-31 (1195 monthly observations).
- Duplicate dates: 0.
- Total missing values after cleaning: 0.
- Highest average monthly percent return in the cleaned dataset: small_hibm_vwret_pct at 1.4203.

The section can be expanded later with richer descriptive plots and discussion, but the current draft already documents the data source, storage conventions, and basic data-quality checks used by the reproducible pipeline.
