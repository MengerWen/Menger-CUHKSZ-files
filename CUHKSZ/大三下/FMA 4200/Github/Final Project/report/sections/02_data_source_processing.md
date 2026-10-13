# Data Source and Processing

## Kenneth French Source

The project uses the local `Data.csv` file supplied with the course. The descriptive header in the raw file states that the file was created using the **202601 CRSP database** and contains returns for portfolios formed on **market equity (ME)** and **book equity to market equity (BE/ME)**. The official Kenneth French Data Library page for the matching dataset describes these as six value-weighted portfolios formed from the intersection of two size groups and three book-to-market groups, rebalanced each June. The local course file covers **July 1926 through January 2026**, which is the sample used in this project.

## Raw File Structure

The raw CSV is not analysis-ready. It contains:

- descriptive text rows before the actual data header,
- the monthly value-weighted return block used in this project,
- and footer text at the end of the file.

The raw file also documents that missing values are encoded as `-99.99` or `-999`. Those sentinel values must therefore be converted during parsing rather than treated as genuine returns.

## Preprocessing and Cleaning

The cleaning pipeline is implemented in `scripts/clean_data.py` and `src/fma4200_project/data_pipeline.py`. The pipeline performs the following steps reproducibly:

1. Programmatically locates the true header row for the monthly value-weighted six-portfolio block.
2. Extracts only the monthly value-weighted section of the file.
3. Converts the raw `YYYYMM` identifier into a proper month-end `date` column.
4. Renames the six portfolio columns into stable snake_case variable names.
5. Converts `-99.99` and `-999` to `NaN`.
6. Saves two cleaned datasets:
   - `data/processed/monthly_portfolio_returns_clean.csv`
   - `data/processed/monthly_portfolio_returns_decimal.csv`
7. Saves data-quality checks, descriptive tables, and baseline figures under `output/`.

## Return Units and Storage Convention

The Kenneth French monthly portfolio file reports returns in **percent units**, not decimal form. For example, the raw value `1.0866` means a monthly return of **1.0866%**, which is equivalent to **0.010866** in decimal form.

To keep the convention explicit:

- the canonical cleaned dataset stores the six portfolio returns in percent units and uses the suffix `_pct`,
- the companion dataset stores decimal returns and uses the suffix `_dec`,
- and the conversion rule is always `decimal_return = percent_return / 100.0`.

This convention is documented in `data/processed/data_dictionary.csv` and `output/tables/sanity_checks_summary.csv`.

## Variable Definitions

Table 1 summarizes the key variables used in the cleaned percent dataset.

| Variable | Interpretation | Unit |
| --- | --- | --- |
| `date` | Month-end calendar date derived from the raw `YYYYMM` code | date |
| `small_lobm_vwret_pct` | Small size, low book-to-market, value-weighted monthly return | percent |
| `me1_bm2_vwret_pct` | Small size, middle book-to-market, value-weighted monthly return | percent |
| `small_hibm_vwret_pct` | Small size, high book-to-market, value-weighted monthly return | percent |
| `big_lobm_vwret_pct` | Big size, low book-to-market, value-weighted monthly return | percent |
| `me2_bm2_vwret_pct` | Big size, middle book-to-market, value-weighted monthly return | percent |
| `big_hibm_vwret_pct` | Big size, high book-to-market, value-weighted monthly return | percent |

The full machine-readable variable descriptions, including the decimal companion variables, are stored in `data/processed/data_dictionary.csv`.

## Sample Period and Data Checks

The cleaned monthly sample runs from **1926-07-31** to **2026-01-31**, for a total of **1195 monthly observations**. The saved data-quality checks confirm:

- `0` duplicate date rows,
- `0` missing months in the monthly sequence,
- and `0` missing values after sentinel conversion.

These checks are saved in:

- `output/tables/date_coverage_check.csv`
- `output/tables/duplicate_check.csv`
- `output/tables/missingness_check.csv`

## Summary Statistics

The basic summary statistics already show economically meaningful cross-portfolio differences. Table 2 reports the monthly mean and volatility along with annualized counterparts.

| Portfolio | Mean (%) | Annualized mean (%) | Std. dev. (%) | Annualized vol. (%) |
| --- | ---: | ---: | ---: | ---: |
| `small_lobm_vwret_pct` | 0.9743 | 11.6917 | 7.4321 | 25.7455 |
| `me1_bm2_vwret_pct` | 1.2361 | 14.8336 | 6.9418 | 24.0471 |
| `small_hibm_vwret_pct` | 1.4203 | 17.0438 | 8.0803 | 27.9910 |
| `big_lobm_vwret_pct` | 0.9582 | 11.4983 | 5.2660 | 18.2421 |
| `me2_bm2_vwret_pct` | 0.9622 | 11.5465 | 5.6016 | 19.4045 |
| `big_hibm_vwret_pct` | 1.2126 | 14.5516 | 7.0808 | 24.5287 |

Two patterns stand out immediately. First, the small high book-to-market portfolio has both the highest average return and the highest volatility, which is broadly consistent with the idea that value and size exposures are compensated but risky. Second, the big low book-to-market portfolio is materially less volatile than the small portfolios, which makes it a natural benchmark when the report later compares risk-adjusted performance and efficient-frontier allocations.

The full descriptive tables are saved in:

- `output/tables/descriptive_statistics_pct.csv`
- `output/tables/descriptive_statistics_decimal.csv`
- `output/tables/portfolio_summary_snapshot.csv`

## Correlation Structure and Informative Figures

The six portfolios are strongly positively correlated, with pairwise correlations ranging from roughly **0.7785** to **0.9614**. This is expected because all six portfolios are exposed to the broad U.S. equity market, but the differences are still wide enough to make joint modeling and diversification analysis meaningful. The weakest correlation appears between the small high book-to-market and big low book-to-market portfolios, while the strongest correlation appears between the middle-small and small-high book-to-market portfolios.

The most useful baseline visuals saved under `output/` for this section are:

- `output/figures/monthly_returns_overview.png`, which shows the time-series behavior of each portfolio return.
- `output/figures/cumulative_growth_of_1.png`, which converts returns into the growth path of a one-dollar investment.
- `output/figures/portfolio_correlation_heatmap.png`, which summarizes the strong but non-identical cross-portfolio dependence.

The supporting table for the correlation discussion is:

- `output/tables/portfolio_correlation_matrix.csv`

## Interpretation for Later Sections

The data-processing stage already points toward the later modeling choices required by the assignment. High cross-portfolio correlation motivates multivariate modeling and relative-value analysis. The visible changes in volatility across historical episodes motivate GARCH-type specifications. The differences in average return and volatility across size and book-to-market groups motivate both predictive modeling and portfolio optimization. In short, the cleaned data are not only ready for analysis; they already provide a coherent empirical reason for the report's later modeling and trading sections.
