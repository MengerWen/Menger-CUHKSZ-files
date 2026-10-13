from __future__ import annotations

import pandas as pd

from funding_arb.reporting.data_quality import (
    compute_correlation_summary,
    compute_missingness_summary,
    compute_time_coverage_summary,
    prepare_analysis_frame,
)


def test_compute_missingness_summary_reports_counts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC"),
            "perp_close": [1.0, None, 3.0],
            "spot_close": [1.0, 2.0, 3.0],
        }
    )
    summary = compute_missingness_summary(frame)
    perp_row = summary.loc[summary["column"] == "perp_close"].iloc[0]
    assert perp_row["missing_count"] == 1
    assert round(float(perp_row["missing_pct"]), 4) == 33.3333


def test_compute_time_coverage_summary_detects_internal_gap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T01:00:00Z"),
                pd.Timestamp("2024-01-01T03:00:00Z"),
            ],
            "funding_event": [0, 1, 0],
        }
    )
    summary = compute_time_coverage_summary(frame, frequency="1h")
    row = summary.iloc[0]
    assert row["missing_hours_inside_observed_range"] == 1
    assert row["non_standard_gap_count"] == 1
    assert row["max_gap_hours"] == 2.0
    assert row["funding_event_rows"] == 1


def test_prepare_analysis_frame_creates_derived_metrics_and_correlations() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC"),
            "perp_close": [100.0, 101.0, 103.0, 102.0, 104.0, 105.0],
            "spot_close": [99.8, 100.7, 102.5, 101.7, 103.4, 104.1],
            "funding_rate": [0.0001, 0.0002, -0.0001, 0.0, 0.00015, 0.00012],
            "perp_volume": [10, 11, 12, 13, 14, 15],
            "spot_volume": [8, 9, 10, 11, 12, 13],
        }
    )
    analysis = prepare_analysis_frame(frame, volatility_window_hours=3, annualization_factor_hours=24 * 365)
    assert "spread_bps" in analysis.columns
    assert "funding_rate_bps" in analysis.columns
    assert "perp_rolling_vol_annualized" in analysis.columns
    correlations = compute_correlation_summary(
        analysis,
        ["perp_return", "spot_return", "funding_rate_bps", "spread_bps", "perp_volume", "spot_volume"],
    )
    assert list(correlations.columns) == ["perp_return", "spot_return", "funding_rate_bps", "spread_bps", "perp_volume", "spot_volume"]
    assert correlations.loc["perp_return", "perp_return"] == 1.0