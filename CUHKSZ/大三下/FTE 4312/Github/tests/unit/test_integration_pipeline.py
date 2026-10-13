from __future__ import annotations

import pandas as pd

from funding_arb.config.models import IntegrationSettings
from funding_arb.integration.pipeline import (
    _build_selected_snapshot,
    _build_update_plan,
    _choose_leaderboard_row,
    _choose_signal_row,
)


def _sample_settings() -> IntegrationSettings:
    return IntegrationSettings.model_validate(
        {
            "selection": {
                "strategy_name": None,
                "ranking_metric": "total_net_pnl_usd",
                "ranking_ascending": False,
                "split_preference": ["test", "validation", "train"],
                "prefer_should_trade": False,
                "require_should_trade": False,
                "allow_flat_fallback": True,
            },
            "contract": {
                "broadcast": False,
                "update_strategy_state": True,
                "update_nav": True,
                "update_pnl": False,
            },
            "semantics": {
                "base_nav_assets": 100_000_000,
                "asset_decimals": 6,
                "asset_usd_price": 1.0,
                "nav_floor_assets": 0,
                "flat_strategy_state": "idle",
                "active_strategy_state": "active",
            },
        }
    )


def test_choose_leaderboard_row_prefers_top_ranked_strategy() -> None:
    settings = _sample_settings()
    signals = pd.DataFrame({"strategy_name": ["alpha", "beta"]})
    leaderboard = pd.DataFrame(
        {
            "strategy_name": ["alpha", "beta"],
            "total_net_pnl_usd": [10.0, 25.0],
        }
    )

    strategy_name, summary = _choose_leaderboard_row(signals, leaderboard, settings)

    assert strategy_name == "beta"
    assert summary["total_net_pnl_usd"] == 25.0


def test_choose_leaderboard_row_prefers_traded_strategy_over_no_trade_zero() -> None:
    settings = _sample_settings()
    signals = pd.DataFrame({"strategy_name": ["no_trade", "traded"]})
    leaderboard = pd.DataFrame(
        {
            "strategy_name": ["no_trade", "traded"],
            "has_trades": [False, True],
            "trade_count": [0, 12],
            "total_net_pnl_usd": [0.0, -50.0],
        }
    )

    strategy_name, summary = _choose_leaderboard_row(signals, leaderboard, settings)

    assert strategy_name == "traded"
    assert summary["trade_count"] == 12


def test_choose_signal_row_respects_split_preference() -> None:
    settings = _sample_settings()
    signals = pd.DataFrame(
        {
            "strategy_name": ["beta", "beta", "beta"],
            "split": ["train", "validation", "test"],
            "should_trade": [0, 1, 0],
            "timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
        }
    )

    signal_row, split_name = _choose_signal_row(signals, "beta", settings)

    assert split_name == "test"
    assert str(signal_row["timestamp"]) == "2024-01-03 00:00:00+00:00"


def test_build_update_plan_uses_active_state_for_tradeable_signal() -> None:
    settings = _sample_settings()
    signal_row = pd.Series(
        {
            "timestamp": pd.Timestamp("2024-01-03T00:00:00Z"),
            "should_trade": 1,
            "suggested_direction": "short_perp_long_spot",
            "signal_score": 0.75,
            "expected_return_bps": 12.5,
            "confidence": 0.9,
            "source": "baseline",
            "source_subtype": "baseline_ml",
            "model_family": "linear",
            "task": "regression",
            "metadata_json": '{"example": true}',
        }
    )
    leaderboard_summary = {
        "strategy_name": "ridge_regression",
        "total_net_pnl_usd": 25.0,
    }

    snapshot = _build_selected_snapshot(
        signal_row,
        "test",
        "ridge_regression",
        leaderboard_summary,
        settings,
    )
    plan = _build_update_plan(snapshot, settings)

    assert plan.strategy_state_name == "active"
    assert plan.strategy_state_code == 1
    assert plan.reported_nav_assets == 125_000_000
    assert plan.summary_pnl_assets == 25_000_000
    assert plan.selected_strategy_name == "ridge_regression"
    assert plan.signal_hash.startswith("0x")
    assert len(plan.signal_hash) == 66
    assert len(plan.metadata_hash) == 66
    assert len(plan.report_hash) == 66
