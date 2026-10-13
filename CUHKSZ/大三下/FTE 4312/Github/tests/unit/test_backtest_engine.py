from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from funding_arb.backtest.engine import (
    _funding_pnl_for_period,
    _leverage_diagnostics,
    _plot_trade_return_boxplot,
    build_mark_to_market_equity_curve,
    build_realized_equity_curve,
    calculate_trade_pnl,
    run_backtest_pipeline,
    summarize_strategy_backtest,
)
from funding_arb.config.models import BacktestSettings


def _settings(**overrides: object) -> BacktestSettings:
    config = {
        "input": {
            "signal_path": "unused.csv",
            "signal_manifest_path": None,
            "market_dataset_path": "unused.csv",
            "market_manifest_path": None,
            "provider": "binance",
            "symbol": "BTCUSDT",
            "venue": "binance",
            "frequency": "1h",
        },
        "selection": {
            "strategy_names": [],
            "split_filter": ["train", "validation", "test"],
            "direction": "short_perp_long_spot",
            "require_should_trade": True,
        },
        "portfolio": {
            "initial_capital": 100_000.0,
            "position_notional": 10_000.0,
            "max_open_positions": 1,
            "max_gross_leverage": 2.0,
            "leverage_check_mode": "warn",
        },
        "costs": {
            "taker_fee_bps": 0.0,
            "maker_fee_bps": 0.0,
            "slippage_bps": 0.0,
            "gas_cost_usd": 0.0,
            "other_friction_bps": 0.0,
        },
        "execution": {
            "entry_delay_bars": 0,
            "execution_price_field": "close",
            "holding_window_hours": 1,
            "maximum_holding_hours": 1,
            "funding_interval_hours": 8,
            "funding_mode": "prototype_bar_sum",
            "funding_notional_mode": "initial_notional",
            "hedge_mode": "equal_notional_hedge",
            "rebalance_frequency": "1h",
            "exit_on_signal_off": False,
            "allow_partial_exit": False,
        },
        "reporting": {
            "output_dir": "tests/.tmp/backtest_outputs",
            "run_name": "unit",
            "write_csv": False,
            "write_markdown_report": False,
            "figure_format": "png",
            "dpi": 72,
            "top_n_strategies_for_plots": 1,
            "primary_split": "test",
            "include_combined_summary": True,
        },
    }
    for section, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(section), dict):
            config[section].update(value)
        else:
            config[section] = value
    return BacktestSettings.model_validate(config)


def test_calculate_trade_pnl_for_short_perp_long_spot_convergence_trade() -> None:
    pnl = calculate_trade_pnl(
        direction="short_perp_long_spot",
        position_notional_usd=10_000.0,
        perp_entry_price_raw=100.0,
        perp_exit_price_raw=99.0,
        spot_entry_price_raw=100.0,
        spot_exit_price_raw=101.0,
        funding_rate_sum=0.001,
        taker_fee_bps=0.0,
        slippage_bps=0.0,
        gas_cost_usd=0.0,
        other_friction_bps=0.0,
    )

    assert pnl["perp_leg_pnl_usd"] == pytest.approx(100.0)
    assert pnl["spot_leg_pnl_usd"] == pytest.approx(100.0)
    assert pnl["funding_pnl_usd"] == pytest.approx(10.0)
    assert pnl["gross_pnl_usd"] == pytest.approx(210.0)
    assert pnl["net_pnl_usd"] == pytest.approx(210.0)
    assert pnl["gross_return_bps"] == pytest.approx(210.0)


def test_calculate_trade_pnl_applies_explicit_cost_terms() -> None:
    pnl = calculate_trade_pnl(
        direction="short_perp_long_spot",
        position_notional_usd=10_000.0,
        perp_entry_price_raw=100.0,
        perp_exit_price_raw=100.0,
        spot_entry_price_raw=100.0,
        spot_exit_price_raw=100.0,
        funding_rate_sum=0.0,
        taker_fee_bps=5.0,
        slippage_bps=0.0,
        gas_cost_usd=2.0,
        other_friction_bps=1.0,
    )

    assert pnl["trading_fees_usd"] == pytest.approx(20.0)
    assert pnl["other_friction_usd"] == pytest.approx(1.0)
    assert pnl["net_pnl_usd"] == pytest.approx(-23.0)
    assert pnl["net_return_bps"] == pytest.approx(-23.0)


def test_trade_pnl_reports_embedded_slippage_without_double_deducting() -> None:
    pnl = calculate_trade_pnl(
        direction="short_perp_long_spot",
        position_notional_usd=10_000.0,
        perp_entry_price_raw=100.0,
        perp_exit_price_raw=100.0,
        spot_entry_price_raw=100.0,
        spot_exit_price_raw=100.0,
        funding_rate_sum=0.0,
        taker_fee_bps=0.0,
        slippage_bps=5.0,
        gas_cost_usd=0.0,
        other_friction_bps=0.0,
    )

    assert pnl["embedded_slippage_cost_usd"] == pytest.approx(pnl["estimated_slippage_cost_usd"])
    assert pnl["net_pnl_usd"] == pytest.approx(pnl["gross_pnl_usd"])


def test_mark_to_market_curve_captures_intratrade_drawdown() -> None:
    timestamps = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "perp_close": [100.0, 110.0, 100.0],
            "spot_close": [100.0, 90.0, 100.0],
            "funding_rate": [0.0, 0.0, 0.0],
        }
    )
    trade_log = pd.DataFrame(
        {
            "entry_market_index": [0],
            "exit_market_index": [2],
            "net_pnl_usd": [0.0],
            "direction": ["short_perp_long_spot"],
            "position_notional_usd": [10_000.0],
            "perp_entry_price_raw": [100.0],
            "spot_entry_price_raw": [100.0],
        }
    )

    curve = build_mark_to_market_equity_curve(
        market,
        trade_log,
        initial_capital=100_000.0,
        strategy_name="toy_strategy",
        settings=_settings(),
        curve_scope="test",
    )

    assert curve["realized_equity_usd"].tolist() == pytest.approx([100_000.0, 100_000.0, 100_000.0])
    assert curve["mark_to_market_equity_usd"].iloc[1] == pytest.approx(98_000.0)
    assert curve["mark_to_market_drawdown"].min() == pytest.approx(-0.02)


def test_funding_mode_event_aware_uses_funding_event_hours() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    market = pd.DataFrame(
        {
            "timestamp": timestamps,
            "perp_close": [100.0] * 10,
            "funding_rate": [0.001] * 10,
        }
    )

    prototype = _funding_pnl_for_period(
        direction="short_perp_long_spot",
        position_notional_usd=10_000.0,
        perp_entry_price_raw=100.0,
        market=market,
        start_index=0,
        end_index=10,
        settings=_settings(execution={"funding_mode": "prototype_bar_sum"}),
    )
    event_aware = _funding_pnl_for_period(
        direction="short_perp_long_spot",
        position_notional_usd=10_000.0,
        perp_entry_price_raw=100.0,
        market=market,
        start_index=0,
        end_index=10,
        settings=_settings(execution={"funding_mode": "event_aware"}),
    )

    assert prototype["funding_rate_sum"] == pytest.approx(0.010)
    assert event_aware["funding_rate_sum"] == pytest.approx(0.002)
    assert prototype["funding_rows_used"] == pytest.approx(10)
    assert event_aware["funding_rows_used"] == pytest.approx(2)


def test_leverage_guard_can_fail_fast() -> None:
    settings = _settings(portfolio={"max_gross_leverage": 0.1, "leverage_check_mode": "fail"})

    with pytest.raises(ValueError, match="gross leverage"):
        _leverage_diagnostics(settings)


def test_realized_equity_curve_and_summary_metrics() -> None:
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    trade_log = pd.DataFrame(
        {
            "entry_timestamp": [timestamps[0], timestamps[2]],
            "exit_market_index": [1, 3],
            "net_pnl_usd": [100.0, -50.0],
            "net_return_bps": [100.0, -50.0],
            "turnover_usd": [40_000.0, 40_000.0],
            "trading_fees_usd": [0.0, 0.0],
            "gas_cost_usd": [0.0, 0.0],
            "other_friction_usd": [0.0, 0.0],
            "funding_pnl_usd": [10.0, -5.0],
            "gross_pnl_usd": [100.0, -50.0],
            "holding_hours": [1, 1],
        }
    )

    equity_curve = build_realized_equity_curve(
        pd.Series(timestamps),
        trade_log,
        initial_capital=100_000.0,
        strategy_name="toy_strategy",
    )
    summary = summarize_strategy_backtest(
        strategy_name="toy_strategy",
        source="baseline",
        source_subtype="rule_based",
        task="classification",
        equity_curve=equity_curve,
        trade_log=trade_log,
        initial_capital=100_000.0,
        strategy_metadata={
            "signal_threshold": 1.5,
            "signal_threshold_mode": "constant",
            "threshold_objective": "avg_signal_return_bps",
            "selected_threshold_objective_value": 0.8,
            "prediction_mode": "static",
            "calibration_method": "none",
            "feature_importance_method": "not_applicable",
            "selected_hyperparameters_json": "{}",
            "checkpoint_selection_metric": "validation_avg_signal_return_bps",
            "best_checkpoint_metric_value": None,
            "checkpoint_selection_effective_metric": "validation_loss",
            "best_checkpoint_effective_metric_value": 0.95,
            "checkpoint_selection_fallback_used": True,
            "selected_loss": "huber",
            "regression_loss": "huber",
            "use_balanced_classification_loss": False,
            "preprocessing_scaler": "robust",
            "winsorize_lower_quantile": 0.01,
            "winsorize_upper_quantile": 0.99,
        },
    )

    assert equity_curve["equity_usd"].tolist() == pytest.approx(
        [100_000.0, 100_100.0, 100_100.0, 100_050.0]
    )
    assert summary["trade_count"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)
    assert summary["total_net_pnl_usd"] == pytest.approx(50.0)
    assert summary["cumulative_return"] == pytest.approx(0.0005)
    assert summary["final_equity_usd"] == pytest.approx(100_050.0)
    assert summary["signal_threshold"] == pytest.approx(1.5)
    assert summary["prediction_mode"] == "static"
    assert summary["threshold_objective"] == "avg_signal_return_bps"
    assert summary["checkpoint_selection_effective_metric"] == "validation_loss"
    assert summary["checkpoint_selection_fallback_used"] is True
    assert summary["selected_loss"] == "huber"
    assert summary["preprocessing_scaler"] == "robust"
    assert summary["evaluation_split"] == "combined"
    assert "mark_to_market_max_drawdown" in summary


def test_backtest_pipeline_uses_test_split_for_primary_leaderboard() -> None:
    scratch_dir = Path("tests/.tmp/backtest_split")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    market_path = scratch_dir / "market.csv"
    signals_path = scratch_dir / "signals.csv"
    timestamps = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 6,
            "venue": ["binance"] * 6,
            "frequency": ["1h"] * 6,
            "perp_open": [100.0, 101.0, 100.0, 100.0, 99.0, 100.0],
            "perp_close": [100.0, 101.0, 100.0, 100.0, 99.0, 100.0],
            "spot_open": [100.0, 99.0, 100.0, 100.0, 101.0, 100.0],
            "spot_close": [100.0, 99.0, 100.0, 100.0, 101.0, 100.0],
            "funding_rate": [0.0] * 6,
        }
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[1], timestamps[3], timestamps[4]],
            "asset": ["BTCUSDT"] * 4,
            "source": ["baseline"] * 4,
            "source_subtype": ["rule_based"] * 4,
            "strategy_name": ["toy_strategy"] * 4,
            "task": ["classification"] * 4,
            "signal_score": [1.0] * 4,
            "predicted_class": [1] * 4,
            "expected_return_bps": [10.0] * 4,
            "suggested_direction": ["short_perp_long_spot"] * 4,
            "confidence": [0.8] * 4,
            "should_trade": [1] * 4,
            "split": ["train", "train", "test", "test"],
            "metadata_json": ["{}"] * 4,
        }
    ).to_csv(signals_path, index=False)
    settings = _settings(
        input={
            "signal_path": str(signals_path),
            "market_dataset_path": str(market_path),
        },
        reporting={
            "output_dir": str(scratch_dir / "artifacts"),
            "run_name": "primary_test",
            "primary_split": "test",
        },
    )

    artifacts = run_backtest_pipeline(settings)
    leaderboard = pd.read_parquet(artifacts.leaderboard_path)
    trade_log = pd.read_parquet(artifacts.trade_log_path)
    primary_trade_log = pd.read_parquet(artifacts.primary_trade_log_path)
    equity_curve = pd.read_parquet(artifacts.equity_curve_path)
    manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))

    assert len(trade_log) == 2
    assert len(primary_trade_log) == 1
    assert leaderboard.loc[0, "evaluation_split"] == "test"
    assert leaderboard.loc[0, "trade_count"] == 1
    assert set(equity_curve["curve_scope"]) == {"test"}
    assert artifacts.combined_strategy_metrics_path is not None
    assert manifest["summary"]["primary_trade_log_rows"] == 1
    assert manifest["artifacts"]["primary_trade_log_path"] == artifacts.primary_trade_log_path


def test_backtest_pipeline_marks_no_tradable_signals_in_leaderboard() -> None:
    scratch_dir = Path("tests/.tmp/backtest_no_signals")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    market_path = scratch_dir / "market.csv"
    signals_path = scratch_dir / "signals.csv"
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 4,
            "venue": ["binance"] * 4,
            "frequency": ["1h"] * 4,
            "perp_open": [100.0] * 4,
            "perp_close": [100.0] * 4,
            "spot_open": [100.0] * 4,
            "spot_close": [100.0] * 4,
            "funding_rate": [0.0] * 4,
        }
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        {
            "timestamp": [timestamps[1], timestamps[2]],
            "asset": ["BTCUSDT"] * 2,
            "source": ["baseline"] * 2,
            "source_subtype": ["baseline_linear"] * 2,
            "strategy_name": ["elastic_net_regression"] * 2,
            "task": ["regression"] * 2,
            "signal_score": [-1.0, -2.0],
            "predicted_class": [0, 0],
            "expected_return_bps": [-1.0, -2.0],
            "suggested_direction": ["flat", "flat"],
            "confidence": [None, None],
            "should_trade": [0, 0],
            "split": ["validation", "test"],
            "metadata_json": ["{}", "{}"],
        }
    ).to_csv(signals_path, index=False)
    settings = _settings(
        input={
            "signal_path": str(signals_path),
            "market_dataset_path": str(market_path),
        },
        reporting={
            "output_dir": str(scratch_dir / "artifacts"),
            "run_name": "no_signal_case",
            "primary_split": "test",
        },
    )

    artifacts = run_backtest_pipeline(settings)
    leaderboard = pd.read_parquet(artifacts.leaderboard_path)

    assert leaderboard.loc[0, "trade_count"] == 0
    assert leaderboard.loc[0, "status"] == "no_tradable_signals"
    assert "signal_count == 0" in leaderboard.loc[0, "diagnostic_reason"]
    assert pd.isna(leaderboard.loc[0, "sharpe_ratio"])


def test_backtest_pipeline_supports_any_direction_for_exploratory_signals() -> None:
    scratch_dir = Path("tests/.tmp/backtest_any_direction")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    market_path = scratch_dir / "market.csv"
    signals_path = scratch_dir / "signals.csv"
    timestamps = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
    pd.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["BTCUSDT"] * 4,
            "venue": ["binance"] * 4,
            "frequency": ["1h"] * 4,
            "perp_open": [100.0, 101.0, 100.0, 100.0],
            "perp_close": [100.0, 101.0, 100.0, 100.0],
            "spot_open": [100.0, 99.0, 100.0, 100.0],
            "spot_close": [100.0, 99.0, 100.0, 100.0],
            "funding_rate": [0.0] * 4,
        }
    ).to_csv(market_path, index=False)
    pd.DataFrame(
        {
            "timestamp": [timestamps[0]],
            "asset": ["BTCUSDT"],
            "source": ["exploratory_dl"],
            "source_subtype": ["deep_learning_showcase"],
            "strategy_name": ["transformer_direction__rolling_top_decile_abs"],
            "task": ["classification"],
            "signal_score": [0.9],
            "predicted_class": [0],
            "expected_return_bps": [4.0],
            "suggested_direction": ["long_perp_short_spot"],
            "confidence": [0.9],
            "should_trade": [1],
            "split": ["test"],
            "metadata_json": ["{}"],
        }
    ).to_csv(signals_path, index=False)
    settings = _settings(
        input={
            "signal_path": str(signals_path),
            "market_dataset_path": str(market_path),
        },
        selection={"direction": "any"},
        reporting={
            "output_dir": str(scratch_dir / "artifacts"),
            "run_name": "any_direction_case",
            "primary_split": "test",
        },
    )

    artifacts = run_backtest_pipeline(settings)
    leaderboard = pd.read_parquet(artifacts.leaderboard_path)

    assert leaderboard.loc[0, "source"] == "exploratory_dl"
    assert leaderboard.loc[0, "trade_count"] == 1
    assert leaderboard.loc[0, "status"] == "completed"


def test_trade_return_boxplot_handles_empty_trade_log() -> None:
    leaderboard = pd.DataFrame({"strategy_name": ["ridge_regression"]})
    scratch_dir = Path("tests/.tmp/backtest")
    scratch_dir.mkdir(parents=True, exist_ok=True)
    output_path = scratch_dir / "boxplot.png"

    written = _plot_trade_return_boxplot(
        pd.DataFrame(),
        leaderboard,
        output_path,
        top_n=1,
        dpi=72,
    )

    assert written.endswith("boxplot.png")
    assert output_path.exists()
