from __future__ import annotations

import pandas as pd

from funding_arb.data.cleaning import (
    align_hourly_market_data,
    drop_duplicates_and_sort,
    resolve_time_range,
)


def test_resolve_time_range_expands_date_only_end_to_next_day() -> None:
    time_range = resolve_time_range("2024-01-01", "2024-01-03")
    assert time_range.start == pd.Timestamp("2024-01-01T00:00:00Z")
    assert time_range.end_exclusive == pd.Timestamp("2024-01-04T00:00:00Z")


def test_drop_duplicates_and_sort_keeps_last_duplicate() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T01:00:00Z"),
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T00:00:00Z"),
            ],
            "close": [101.0, 99.0, 100.0],
        }
    )
    cleaned = drop_duplicates_and_sort(frame)
    assert list(cleaned["close"]) == [100.0, 101.0]


def test_align_hourly_market_data_fills_prices_and_funding_schedule() -> None:
    perpetual = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T02:00:00Z"),
            ],
            "open": [100.0, 102.0],
            "high": [101.0, 103.0],
            "low": [99.0, 101.0],
            "close": [100.5, 102.5],
            "volume": [10.0, 11.0],
        }
    )
    spot = pd.DataFrame(
        {
            "timestamp": [
                pd.Timestamp("2024-01-01T00:00:00Z"),
                pd.Timestamp("2024-01-01T02:00:00Z"),
            ],
            "open": [99.5, 101.5],
            "high": [100.5, 102.5],
            "low": [98.5, 100.5],
            "close": [100.0, 102.0],
            "volume": [20.0, 21.0],
        }
    )
    funding = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01T00:00:00Z")],
            "funding_rate": [0.0001],
        }
    )

    aligned = align_hourly_market_data(
        perpetual,
        spot,
        funding,
        time_range=resolve_time_range("2024-01-01", "2024-01-01T03:00:00Z"),
        symbol="BTCUSDT",
        venue="binance",
        frequency="1h",
        max_forward_fill_hours=2,
        fill_volume_value=0.0,
        fill_funding_value=0.0,
    )

    assert list(aligned["timestamp"]) == [
        pd.Timestamp("2024-01-01T00:00:00Z"),
        pd.Timestamp("2024-01-01T01:00:00Z"),
        pd.Timestamp("2024-01-01T02:00:00Z"),
    ]
    assert aligned.loc[1, "perp_close"] == 100.5
    assert aligned.loc[1, "spot_close"] == 100.0
    assert aligned.loc[1, "funding_rate"] == 0.0
    assert aligned.loc[0, "funding_event"] == 1
    assert aligned.loc[1, "funding_event"] == 0
    assert bool(aligned.loc[1, "perp_close_was_missing"]) is True