from __future__ import annotations

import logging
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (
    AUTOGEN_DATA_SNAPSHOT_PATH,
    CLEAN_DECIMAL_DATA_PATH,
    CLEAN_PERCENT_DATA_PATH,
    CORRELATION_HEATMAP_PATH,
    CORRELATION_MATRIX_PATH,
    DATA_DICTIONARY_PATH,
    DATE_COVERAGE_PATH,
    DECIMAL_COLUMNS,
    DESCRIPTIVE_STATS_DEC_PATH,
    DESCRIPTIVE_STATS_PCT_PATH,
    DUPLICATE_CHECK_PATH,
    FIGURES_DIR,
    GROWTH_FIGURE_PATH,
    LOGS_DIR,
    MISSINGNESS_PATH,
    MODELS_README_PATH,
    PERCENT_COLUMNS,
    PIPELINE_LOG_PATH,
    PROJECT_DIRECTORIES,
    RAW_DATA_PATH,
    RAW_TO_PERCENT_COLUMNS,
    RETURNS_FIGURE_PATH,
    SANITY_SUMMARY_PATH,
    SUMMARY_SNAPSHOT_PATH,
    TABLES_DIR,
)


LOGGER_NAME = "fma4200_project"


def configure_logging() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(PIPELINE_LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def ensure_directories() -> None:
    for directory in PROJECT_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)

    model_note = (
        "# Model Output Folder\n\n"
        "This folder is reserved for saved estimation results, fitted objects, and strategy outputs "
        "from later stages of the project.\n"
    )
    MODELS_README_PATH.write_text(model_note, encoding="utf-8")


def extract_monthly_value_weighted_data() -> pd.DataFrame:
    lines = RAW_DATA_PATH.read_text(encoding="utf-8").splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.strip().startswith(",SMALL LoBM")
    )

    block_lines = [lines[header_index]]
    for line in lines[header_index + 1 :]:
        if line.startswith("Copyright"):
            break
        if line.replace(",", "").strip() == "":
            break
        block_lines.append(line)

    raw_block = pd.read_csv(StringIO("\n".join(block_lines)))
    renamed = raw_block.rename(columns={raw_block.columns[0]: "yyyymm", **RAW_TO_PERCENT_COLUMNS})
    renamed = renamed.replace({-99.99: np.nan, -999.0: np.nan})
    renamed["yyyymm"] = renamed["yyyymm"].astype(int)
    renamed["date"] = pd.to_datetime(renamed["yyyymm"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)

    cleaned = renamed.loc[:, ["date", *PERCENT_COLUMNS]].sort_values("date").reset_index(drop=True)
    return cleaned


def build_decimal_dataset(percent_df: pd.DataFrame) -> pd.DataFrame:
    decimal_df = percent_df.copy()
    decimal_df["date"] = pd.to_datetime(decimal_df["date"])
    rename_map = dict(zip(PERCENT_COLUMNS, DECIMAL_COLUMNS))
    decimal_df = decimal_df.rename(columns=rename_map)
    for column in DECIMAL_COLUMNS:
        decimal_df[column] = decimal_df[column] / 100.0
    return decimal_df


def build_data_dictionary() -> pd.DataFrame:
    records = [
        {
            "variable_name": "date",
            "dataset_name": CLEAN_PERCENT_DATA_PATH.name,
            "dtype": "date",
            "unit": "month-end calendar date",
            "description": "Month-end timestamp derived from the YYYYMM code in the raw CSV.",
            "conversion_note": "Converted from raw YYYYMM integer code to ISO date.",
        }
    ]

    portfolio_descriptions = {
        "small_lobm_vwret_pct": "Small size, low book-to-market, value-weighted monthly return.",
        "me1_bm2_vwret_pct": "Small size, middle book-to-market, value-weighted monthly return.",
        "small_hibm_vwret_pct": "Small size, high book-to-market, value-weighted monthly return.",
        "big_lobm_vwret_pct": "Big size, low book-to-market, value-weighted monthly return.",
        "me2_bm2_vwret_pct": "Big size, middle book-to-market, value-weighted monthly return.",
        "big_hibm_vwret_pct": "Big size, high book-to-market, value-weighted monthly return.",
    }

    for column in PERCENT_COLUMNS:
        records.append(
            {
                "variable_name": column,
                "dataset_name": CLEAN_PERCENT_DATA_PATH.name,
                "dtype": "float",
                "unit": "percent monthly return",
                "description": portfolio_descriptions[column],
                "conversion_note": "Stored in percent units exactly as in the monthly value-weighted section of the raw file.",
            }
        )

    for percent_column, decimal_column in zip(PERCENT_COLUMNS, DECIMAL_COLUMNS):
        records.append(
            {
                "variable_name": decimal_column,
                "dataset_name": CLEAN_DECIMAL_DATA_PATH.name,
                "dtype": "float",
                "unit": "decimal monthly return",
                "description": portfolio_descriptions[percent_column].replace("return.", "return in decimal form."),
                "conversion_note": f"Computed as {percent_column} / 100.0.",
            }
        )

    return pd.DataFrame.from_records(records)


def build_date_coverage_check(percent_df: pd.DataFrame) -> pd.DataFrame:
    expected_dates = pd.date_range(percent_df["date"].min(), percent_df["date"].max(), freq="ME")
    actual_dates = pd.DatetimeIndex(percent_df["date"])
    missing_dates = expected_dates.difference(actual_dates)

    coverage = pd.DataFrame(
        {
            "metric": [
                "sample_start",
                "sample_end",
                "n_rows",
                "expected_months_between_start_and_end",
                "duplicate_date_rows",
                "missing_months_in_sequence",
                "frequency_check_passed",
            ],
            "value": [
                actual_dates.min().date().isoformat(),
                actual_dates.max().date().isoformat(),
                len(percent_df),
                len(expected_dates),
                int(percent_df["date"].duplicated().sum()),
                len(missing_dates),
                bool(len(percent_df) == len(expected_dates) and len(missing_dates) == 0),
            ],
        }
    )
    return coverage


def build_duplicate_check(percent_df: pd.DataFrame) -> pd.DataFrame:
    duplicates = percent_df[percent_df["date"].duplicated(keep=False)].copy()
    if duplicates.empty:
        return pd.DataFrame(
            {
                "status": ["passed"],
                "duplicate_rows": [0],
                "note": ["No duplicate dates were found in the cleaned dataset."],
            }
        )
    return duplicates.sort_values("date").reset_index(drop=True)


def build_missingness_check(percent_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "column": percent_df.columns,
            "missing_count": [int(percent_df[column].isna().sum()) for column in percent_df.columns],
            "missing_share": [float(percent_df[column].isna().mean()) for column in percent_df.columns],
        }
    )


def build_descriptive_statistics(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    stats = pd.DataFrame(
        {
            "count": df[value_columns].count(),
            "mean": df[value_columns].mean(),
            "std": df[value_columns].std(),
            "min": df[value_columns].min(),
            "p25": df[value_columns].quantile(0.25),
            "median": df[value_columns].median(),
            "p75": df[value_columns].quantile(0.75),
            "max": df[value_columns].max(),
        }
    )
    stats.index.name = "column"
    return stats.reset_index()


def build_summary_snapshot(percent_df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "portfolio": PERCENT_COLUMNS,
            "mean_pct": percent_df[PERCENT_COLUMNS].mean().values,
            "annualized_mean_pct": (percent_df[PERCENT_COLUMNS].mean() * 12.0).values,
            "std_pct": percent_df[PERCENT_COLUMNS].std().values,
            "annualized_vol_pct": (percent_df[PERCENT_COLUMNS].std() * np.sqrt(12.0)).values,
            "min_pct": percent_df[PERCENT_COLUMNS].min().values,
            "max_pct": percent_df[PERCENT_COLUMNS].max().values,
        }
    )
    return summary


def build_correlation_matrix(percent_df: pd.DataFrame) -> pd.DataFrame:
    return percent_df[PERCENT_COLUMNS].corr()


def build_sanity_summary(
    percent_df: pd.DataFrame,
    date_coverage: pd.DataFrame,
    duplicate_check: pd.DataFrame,
    missingness: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "check": [
                "date_coverage",
                "duplicate_dates",
                "missing_values",
                "raw_unit_convention",
                "clean_storage_convention",
                "decimal_conversion",
            ],
            "result": [
                date_coverage.loc[date_coverage["metric"] == "frequency_check_passed", "value"].iloc[0],
                "passed" if "status" in duplicate_check.columns else "failed",
                "passed" if int(missingness["missing_count"].sum()) == 0 else "review_needed",
                "monthly returns in percent",
                "cleaned percent dataset saved with _pct suffix",
                "decimal companion dataset saved with _dec suffix and values divided by 100",
            ],
            "detail": [
                f"{len(percent_df)} rows from {percent_df['date'].min():%Y-%m-%d} to {percent_df['date'].max():%Y-%m-%d}",
                duplicate_check.iloc[0]["note"] if "status" in duplicate_check.columns else "Duplicate dates detected.",
                f"Total missing cells: {int(missingness['missing_count'].sum())}",
                "The Kenneth French monthly value-weighted section reports returns in percentage points.",
                "Example: 1.0866 means 1.0866%, not 108.66%.",
                "Example: 1.0866% becomes 0.010866 in the decimal companion file.",
            ],
        }
    )


def save_figures(percent_df: pd.DataFrame, decimal_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()

    for axis, column in zip(axes, PERCENT_COLUMNS):
        axis.plot(percent_df["date"], percent_df[column], linewidth=0.9)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_title(column)
        axis.set_ylabel("Return (%)")

    fig.suptitle("Monthly Value-Weighted Portfolio Returns", fontsize=14)
    fig.tight_layout()
    fig.savefig(RETURNS_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)

    cumulative_growth = (1.0 + decimal_df[DECIMAL_COLUMNS]).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    for column in DECIMAL_COLUMNS:
        ax.plot(decimal_df["date"], cumulative_growth[column], linewidth=1.1, label=column)
    ax.set_title("Growth of $1 Using Decimal Returns")
    ax.set_ylabel("Portfolio Value")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(GROWTH_FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix.values, cmap="Blues", vmin=0.75, vmax=1.0)
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    ax.set_title("Portfolio Return Correlations")

    for row in range(correlation_matrix.shape[0]):
        for column in range(correlation_matrix.shape[1]):
            ax.text(
                column,
                row,
                f"{correlation_matrix.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CORRELATION_HEATMAP_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_autogen_snapshot(percent_df: pd.DataFrame, missingness: pd.DataFrame, descriptive_stats: pd.DataFrame) -> None:
    top_mean_row = descriptive_stats.sort_values("mean", ascending=False).iloc[0]
    report_lines = [
        "# Auto-Generated Data Snapshot",
        "",
        "This file is generated by the cleaning pipeline and is meant to support the manually written report sections.",
        "",
        f"- Sample coverage: {percent_df['date'].min():%Y-%m-%d} to {percent_df['date'].max():%Y-%m-%d} ({len(percent_df)} monthly observations).",
        f"- Duplicate dates: {int(percent_df['date'].duplicated().sum())}.",
        f"- Total missing values after cleaning: {int(missingness['missing_count'].sum())}.",
        f"- Highest average monthly percent return in the cleaned dataset: {top_mean_row['column']} at {top_mean_row['mean']:.4f}.",
    ]
    AUTOGEN_DATA_SNAPSHOT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


def run_cleaning_pipeline() -> dict[str, object]:
    ensure_directories()
    logger = configure_logging()
    logger.info("Starting cleaning pipeline.")

    percent_df = extract_monthly_value_weighted_data()
    percent_df["date"] = pd.to_datetime(percent_df["date"])
    decimal_df = build_decimal_dataset(percent_df)
    data_dictionary = build_data_dictionary()
    date_coverage = build_date_coverage_check(percent_df)
    duplicate_check = build_duplicate_check(percent_df)
    missingness = build_missingness_check(percent_df)
    descriptive_stats_pct = build_descriptive_statistics(percent_df, PERCENT_COLUMNS)
    descriptive_stats_dec = build_descriptive_statistics(decimal_df, DECIMAL_COLUMNS)
    summary_snapshot = build_summary_snapshot(percent_df)
    correlation_matrix = build_correlation_matrix(percent_df)
    sanity_summary = build_sanity_summary(percent_df, date_coverage, duplicate_check, missingness)

    percent_df.to_csv(CLEAN_PERCENT_DATA_PATH, index=False)
    decimal_df.to_csv(CLEAN_DECIMAL_DATA_PATH, index=False)
    data_dictionary.to_csv(DATA_DICTIONARY_PATH, index=False)
    date_coverage.to_csv(DATE_COVERAGE_PATH, index=False)
    duplicate_check.to_csv(DUPLICATE_CHECK_PATH, index=False)
    missingness.to_csv(MISSINGNESS_PATH, index=False)
    descriptive_stats_pct.to_csv(DESCRIPTIVE_STATS_PCT_PATH, index=False)
    descriptive_stats_dec.to_csv(DESCRIPTIVE_STATS_DEC_PATH, index=False)
    summary_snapshot.round(6).to_csv(SUMMARY_SNAPSHOT_PATH, index=False)
    correlation_matrix.round(6).to_csv(CORRELATION_MATRIX_PATH)
    sanity_summary.to_csv(SANITY_SUMMARY_PATH, index=False)
    save_figures(percent_df, decimal_df)
    save_correlation_heatmap(correlation_matrix)
    write_autogen_snapshot(percent_df, missingness, descriptive_stats_pct)

    logger.info("Cleaning pipeline completed successfully.")
    logger.info("Saved cleaned data to %s", CLEAN_PERCENT_DATA_PATH)
    logger.info("Saved sanity checks to %s", TABLES_DIR)
    logger.info("Saved figures to %s", FIGURES_DIR)

    return {
        "n_rows": len(percent_df),
        "start_date": percent_df["date"].min(),
        "end_date": percent_df["date"].max(),
        "duplicate_date_rows": int(percent_df["date"].duplicated().sum()),
        "total_missing_values": int(missingness["missing_count"].sum()),
    }
