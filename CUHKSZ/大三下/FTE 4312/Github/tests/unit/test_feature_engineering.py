from __future__ import annotations

import numpy as np
import pandas as pd

from funding_arb.config.models import FeatureSettings
from funding_arb.features.pipeline import build_feature_table
from funding_arb.features.transforms import rolling_zscore, sign_reversal_indicator


def _make_feature_settings() -> FeatureSettings:
    return FeatureSettings.model_validate(
        {
            "input": {
                "dataset_path": "data/processed/binance/btcusdt/1h/hourly_market_data.parquet",
                "provider": "binance",
                "symbol": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
            },
            "feature_set": {
                "rolling_windows": [4, 8],
                "volatility_window": 4,
                "zscore_window": 8,
                "funding_mean_window": 4,
                "basis_mean_window": 4,
                "shock_window": 4,
                "liquidity_window": 4,
                "regime_window": 8,
                "funding_interval_hours": 8,
                "annualization_factor_hours": 24 * 365,
            },
            "labels": {
                "forward_horizon_hours": 8,
                "min_expected_edge_bps": 5,
                "use_post_cost_target": True,
            },
            "output": {
                "processed_dir": "data/processed/features",
                "artifact_name": "test_feature_set.parquet",
                "manifest_name": "test_feature_manifest.json",
                "write_csv": True,
            },
        }
    )


def _make_market_frame(rows: int = 32) -> pd.DataFrame:
    index = np.arange(rows)
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    perp_close = 100.0 + 0.6 * index + np.sin(index / 2.0)
    spot_close = 99.7 + 0.58 * index + np.cos(index / 3.0) * 0.4
    funding_event = (index % 8 == 0).astype(int)
    funding_rate = np.where(funding_event == 1, 0.0001 + 0.00002 * np.sin(index), 0.0)
    perp_volume = 1_000.0 + 15.0 * index + 30.0 * np.sin(index / 4.0)
    spot_volume = 900.0 + 12.0 * index + 20.0 * np.cos(index / 5.0)
    open_interest = 5_000.0 + 25.0 * index
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": "BTCUSDT",
            "venue": "binance",
            "frequency": "1h",
            "perp_close": perp_close,
            "spot_close": spot_close,
            "funding_rate": funding_rate,
            "funding_event": funding_event,
            "perp_volume": perp_volume,
            "spot_volume": spot_volume,
            "open_interest": open_interest,
            "perp_close_was_missing": False,
            "spot_close_was_missing": False,
            "open_interest_was_missing": False,
        }
    )


def test_rolling_zscore_is_not_affected_by_future_values() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    baseline = rolling_zscore(series, window=3)
    modified = rolling_zscore(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 999.0]), window=3)
    assert baseline.iloc[4] == modified.iloc[4]


def test_sign_reversal_indicator_only_flags_nonzero_flips() -> None:
    series = pd.Series([0.0, 0.1, -0.2, -0.3, 0.0, 0.4, -0.5])
    reversal = sign_reversal_indicator(series)
    assert list(reversal.astype(int)) == [0, 0, 1, 0, 0, 0, 1]


def test_build_feature_table_creates_grouped_columns_and_readiness() -> None:
    frame = _make_market_frame()
    settings = _make_feature_settings()
    feature_table, groups = build_feature_table(frame, settings)

    expected_columns = {
        "funding_rate_raw",
        "funding_zscore_8h",
        "spread_bps",
        "spread_reversion_signal_8h",
        "perp_realized_vol_4h",
        "perp_return_shock_4h",
        "perp_volume_ratio_4h",
        "open_interest_zscore_4h",
        "funding_x_spread_bps",
        "high_vol_regime",
        "feature_ready",
    }
    assert expected_columns.issubset(feature_table.columns)
    assert set(groups) == {"funding", "basis", "volatility", "liquidity", "interaction_state"}
    assert feature_table["feature_ready"].iloc[:8].sum() < 8
    assert feature_table["feature_ready"].iloc[12:].sum() > 0
    assert feature_table.columns.duplicated().sum() == 0