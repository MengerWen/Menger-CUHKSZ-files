# Data Source and Processing

## Data Source

This project uses the local `Data.csv` file, which records monthly value-weighted returns for six Kenneth French portfolios sorted by market equity (ME) and book-to-market equity (BE/ME). The raw file states that the portfolios include utilities and financials and were created from the 202601 CRSP database.

## Processing Workflow

The raw CSV contains a descriptive metadata block, one monthly return section, and a footer. The ingestion script `scripts/build_project_baseline.py` automatically locates the true header row instead of relying on a fixed skip count. It then:

1. extracts the monthly return block only,
2. converts the `YYYYMM` code into a month-end calendar date,
3. renames the six portfolio columns to snake_case labels for downstream analysis, and
4. converts the documented missing-value sentinels (`-99.99` and `-999`) to `NaN`.

The cleaned datasets are saved in both wide and tidy forms under `data/processed/`.

## Dataset Snapshot

The cleaned sample runs from **July 1926** to **January 2026**, giving **1195 monthly observations** for each of the six portfolios. The extracted monthly block contains **0 missing values** after sentinel conversion.

From the baseline summary tables:

- `small_hibm` has the highest average monthly return at **1.420%**.
- `small_hibm` has the highest monthly volatility at **8.080%**.
- The strongest pairwise correlation is **me1_bm2 vs small_hibm = 0.961**.
- The weakest pairwise correlation is **small_hibm vs big_lobm = 0.778**.

The main outputs for this section are:

- `outputs/tables/dataset_overview.csv`
- `outputs/tables/summary_statistics.csv`
- `outputs/tables/missingness_check.csv`
- `outputs/tables/correlation_matrix.csv`
- `outputs/figures/monthly_returns.png`
- `outputs/figures/cumulative_growth.png`

These files provide a clean starting point for the later univariate modeling, multivariate modeling, and portfolio strategy sections of the project.
