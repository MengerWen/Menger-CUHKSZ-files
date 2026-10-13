from __future__ import annotations

import pandas as pd

from funding_arb.config.models import LabelPipelineSettings
from funding_arb.labels.generator import assign_time_series_split, build_label_table, forward_window_sum
from funding_arb.labels.pipeline import build_supervised_dataset


def _make_label_settings() -> LabelPipelineSettings:
    return LabelPipelineSettings.model_validate(
        {
            "input": {
                "feature_table_path": "data/processed/features/binance/btcusdt/1h/btcusdt_feature_set.parquet",
                "market_dataset_path": "data/processed/binance/btcusdt/1h/hourly_market_data.parquet",
                "provider": "binance",
                "symbol": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
            },
            "target": {
                "direction": "short_perp_long_spot",
                "holding_windows_hours": [2],
                "primary_horizon_hours": 2,
                "execution_delay_bars": 1,
                "execution_price_field": "open",
                "min_expected_edge_bps": 5.0,
                "positive_return_threshold_bps": 0.0,
                "use_post_cost_target": True,
            },
            "costs": {
                "taker_fee_bps": 1.0,
                "maker_fee_bps": 0.0,
                "slippage_bps": 0.5,
                "gas_cost_usd": 1.0,
                "position_notional_usd": 10000.0,
                "other_friction_bps": 0.0,
                "borrow_cost_bps_per_hour": 0.0,
            },
            "split": {
                "train_end": "2024-01-02",
                "validation_end": "2024-01-04",
                "test_end": "2024-01-06",
            },
            "output": {
                "output_dir": "data/processed/supervised",
                "artifact_name": "test_supervised.parquet",
                "label_table_name": "test_labels.parquet",
                "manifest_name": "test_manifest.json",
                "write_csv": True,
                "save_split_files": True,
            },
        }
    )


def _make_market_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=7, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "perp_open": [100.0, 110.0, 108.0, 106.0, 104.0, 103.0, 102.0],
            "spot_open": [100.0, 100.0, 102.0, 104.0, 105.0, 106.0, 107.0],
            "funding_rate": [0.0, 0.0002, 0.0001, 0.0, 0.0, 0.0, 0.0],
        }
    )


def _make_feature_table() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=7, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "feature_ready": [0, 0, 1, 1, 1, 1, 1],
            "funding_rate_bps": [0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        }
    )


def test_forward_window_sum_only_uses_requested_future_window() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    baseline = forward_window_sum(series, start_offset=1, window=2)
    modified = forward_window_sum(pd.Series([1.0, 2.0, 3.0, 999.0, 5.0]), start_offset=1, window=2)
    assert baseline.iloc[0] == modified.iloc[0]


def test_build_label_table_computes_post_cost_targets_without_leakage() -> None:
    settings = _make_label_settings()
    labels = build_label_table(_make_market_frame(), settings.target, settings.costs)
    row0 = labels.iloc[0]

    expected_price_bps = (-(106.0 / 110.0 - 1.0) + (104.0 / 100.0 - 1.0)) * 10_000.0
    expected_funding_bps = (0.0002 + 0.0001) * 10_000.0
    expected_cost_bps = 4.0 * (1.0 + 0.5) + (1.0 / 10000.0 * 10000.0)
    expected_net_bps = expected_price_bps + expected_funding_bps - expected_cost_bps

    assert round(float(row0["target_future_funding_return_bps_2h"]), 6) == round(expected_funding_bps, 6)
    assert round(float(row0["target_estimated_cost_bps_2h"]), 6) == round(expected_cost_bps, 6)
    assert round(float(row0["target_future_net_return_bps_2h"]), 6) == round(expected_net_bps, 6)
    assert int(row0["target_is_profitable_2h"]) == 1
    assert int(row0["target_is_tradeable_2h"]) == 1


def test_assign_time_series_split_is_chronological() -> None:
    timestamps = pd.Series(pd.date_range("2024-01-01", periods=6, freq="D", tz="UTC"))
    split = assign_time_series_split(timestamps, _make_label_settings().split)
    assert list(split) == ["train", "train", "validation", "validation", "test", "test"]


def test_build_supervised_dataset_marks_modeling_ready_rows() -> None:
    settings = _make_label_settings()
    supervised = build_supervised_dataset(_make_feature_table(), build_label_table(_make_market_frame(), settings.target, settings.costs), settings)
    assert "target_future_net_return_bps_2h" in supervised.columns
    assert "split" in supervised.columns
    assert "supervised_ready" in supervised.columns
    assert supervised.loc[2, "supervised_ready"] == 1
    assert supervised.loc[0, "supervised_ready"] == 0