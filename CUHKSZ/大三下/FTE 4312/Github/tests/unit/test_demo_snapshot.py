from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from funding_arb.demo.pipeline import export_demo_snapshot

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = REPO_ROOT / "tests" / ".tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_temp_dir() -> Path:
    path = TEST_TMP_ROOT / f"demo-snapshot-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_export_demo_snapshot_survives_missing_optional_inputs() -> None:
    tmp_path = _make_temp_dir()
    artifact_dir = tmp_path / "artifacts"
    public_dir = tmp_path / "public"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    data_manifest_path = input_dir / "manifest.json"
    data_quality_summary_path = input_dir / "summary.json"
    backtest_manifest_path = input_dir / "backtest_manifest.json"
    baseline_leaderboard_path = input_dir / "baseline_leaderboard.parquet"
    backtest_leaderboard_path = input_dir / "backtest_leaderboard.parquet"
    chart_path = input_dir / "funding.png"

    _write_json(
        data_manifest_path,
        {
            "dataset": {"symbol": "BTCUSDT", "venue": "binance", "frequency": "1h"},
            "time_range": {
                "start": "2021-01-01T00:00:00+00:00",
                "end_exclusive": "2026-04-07T00:00:00+00:00",
            },
            "canonical_row_count": 100,
            "row_counts": {"perpetual_bars": 100},
        },
    )
    _write_json(
        data_quality_summary_path,
        {
            "funding_event_count": 12,
            "coverage": {"coverage_ratio": 1.0},
            "funding_mean_bps": 0.8,
            "funding_std_bps": 1.4,
            "spread_mean_bps": 2.1,
            "mean_perp_annualized_vol": 0.52,
        },
    )
    _write_json(
        backtest_manifest_path,
        {
            "summary": {
                "strategy_count": 1,
                "trade_count": 4,
                "primary_split": "test",
                "primary_trade_count": 4,
                "combined_trade_count": 4,
                "best_strategy": "ridge_regression",
            },
            "diagnostics": {
                "leverage": {"implied_gross_leverage": 0.2},
                "funding": {"funding_mode": "prototype_bar_sum"},
            },
            "assumptions": ["Prototype assumption one", "Prototype assumption two"],
        },
    )

    pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "ridge_regression",
                "task": "regression",
                "pearson_corr": 0.61,
                "rmse": 2.4,
                "cumulative_signal_return_bps": 18.0,
            }
        ]
    ).to_parquet(baseline_leaderboard_path, index=False)

    pd.DataFrame(
        [
            {
                "strategy_name": "ridge_regression",
                "source": "baseline",
                "source_subtype": "linear",
                "task": "regression",
                "evaluation_split": "test",
                "has_trades": True,
                "trade_count": 4,
                "cumulative_return": 0.012,
                "annualized_return": 0.03,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.01,
                "mark_to_market_max_drawdown": -0.01,
                "realized_max_drawdown": -0.005,
                "win_rate": 0.5,
                "average_trade_return_bps": 6.0,
                "total_net_pnl_usd": 24.5,
                "final_equity_usd": 10024.5,
            }
        ]
    ).to_parquet(backtest_leaderboard_path, index=False)

    chart_path.write_bytes(b"fake-chart")

    config = {
        "demo": {
            "title": "Demo",
            "subtitle": "Fallback-safe dashboard",
            "artifact_dir": str(artifact_dir),
            "frontend_public_dir": str(public_dir),
            "top_strategies": 3,
        },
        "contract": {
            "chain_name": "local_foundry",
            "asset_symbol": "mUSDC",
            "asset_decimals": 6,
            "demo_wallet_cash_assets": 1_000_000_000,
        },
        "inputs": {
            "data_manifest_path": str(data_manifest_path),
            "data_quality_summary_path": str(data_quality_summary_path),
            "backtest_manifest_path": str(backtest_manifest_path),
            "baseline_leaderboard_path": str(baseline_leaderboard_path),
            "dl_leaderboard_path": str(input_dir / "missing_dl.parquet"),
            "backtest_leaderboard_path": str(backtest_leaderboard_path),
            "integration_selection_path": str(input_dir / "missing_selection.json"),
            "integration_plan_path": str(input_dir / "missing_plan.json"),
            "integration_call_summary_path": str(input_dir / "missing_calls.json"),
            "charts": [
                {
                    "title": "Funding",
                    "subtitle": "Existing chart",
                    "section": "data",
                    "source_path": str(chart_path),
                    "target_name": "funding.png",
                },
                {
                    "title": "Missing",
                    "subtitle": "This file is absent",
                    "section": "data",
                    "source_path": str(input_dir / "missing_chart.png"),
                    "target_name": "missing_chart.png",
                },
            ],
        },
    }

    try:
        artifacts = export_demo_snapshot(config)

        snapshot_payload = json.loads(
            Path(artifacts.artifact_snapshot_path).read_text(encoding="utf-8")
        )
        assert Path(artifacts.public_snapshot_path).exists()
        assert (
            snapshot_payload["models"]["deep_learning_best"]["model_name"]
            == "Deep learning not available"
        )
        assert snapshot_payload["vault"]["execution_summary"]["mode"] == "not_run"
        assert snapshot_payload["vault"]["selected_strategy"] == "ridge_regression"
        assert (
            snapshot_payload["backtest"]["best_strategy"]["strategy_name"]
            == "ridge_regression"
        )
        assert snapshot_payload["backtest"]["risk_view"]["primary_split"] == "test"
        assert snapshot_payload["backtest"]["diagnostics"]["funding"]["funding_mode"] == "prototype_bar_sum"
        assert len(snapshot_payload["charts"]) == 1
        assert snapshot_payload["charts"][0]["image"] == "demo/assets/funding.png"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_export_demo_snapshot_prefers_dl_comparison_when_available() -> None:
    tmp_path = _make_temp_dir()
    artifact_dir = tmp_path / "artifacts"
    public_dir = tmp_path / "public"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    data_manifest_path = input_dir / "manifest.json"
    data_quality_summary_path = input_dir / "summary.json"
    backtest_manifest_path = input_dir / "backtest_manifest.json"
    baseline_leaderboard_path = input_dir / "baseline_leaderboard.parquet"
    dl_leaderboard_path = input_dir / "dl_leaderboard.parquet"
    dl_comparison_manifest_path = input_dir / "comparison_manifest.json"
    dl_comparison_summary_path = input_dir / "comparison_summary.parquet"
    dl_comparison_test_path = input_dir / "test_leaderboard.parquet"
    backtest_leaderboard_path = input_dir / "backtest_leaderboard.parquet"

    _write_json(
        data_manifest_path,
        {
            "dataset": {"symbol": "BTCUSDT", "venue": "binance", "frequency": "1h"},
            "time_range": {
                "start": "2021-01-01T00:00:00+00:00",
                "end_exclusive": "2026-04-07T00:00:00+00:00",
            },
            "canonical_row_count": 100,
            "row_counts": {"perpetual_bars": 100},
        },
    )
    _write_json(
        data_quality_summary_path,
        {
            "funding_event_count": 12,
            "coverage": {"coverage_ratio": 1.0},
            "funding_mean_bps": 0.8,
            "funding_std_bps": 1.4,
            "spread_mean_bps": 2.1,
            "mean_perp_annualized_vol": 0.52,
        },
    )
    _write_json(
        backtest_manifest_path,
        {
            "summary": {
                "strategy_count": 1,
                "trade_count": 4,
                "primary_split": "test",
                "primary_trade_count": 4,
                "combined_trade_count": 4,
                "best_strategy": "ridge_regression",
            },
            "diagnostics": {"leverage": {}, "funding": {}},
            "assumptions": ["Prototype assumption one"],
        },
    )
    _write_json(
        dl_comparison_manifest_path,
        {
            "best_model_note": "Transformer wins under test pearson_corr.",
            "run_count": 4,
            "report_path": "comparison_report.md",
        },
    )

    pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "ridge_regression",
                "task": "regression",
                "pearson_corr": 0.61,
                "rmse": 2.4,
            }
        ]
    ).to_parquet(baseline_leaderboard_path, index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "lstm",
                "task": "regression",
                "pearson_corr": 0.62,
                "rmse": 2.2,
            }
        ]
    ).to_parquet(dl_leaderboard_path, index=False)
    comparison_rows = pd.DataFrame(
        [
            {
                "rank": 1,
                "run_label": "transformer_encoder",
                "model_name": "transformer_encoder",
                "model_group": "attention",
                "task": "regression",
                "lookback_steps": 48,
                "ranking_metric": "pearson_corr",
                "ranking_metric_value": 0.67,
                "test_pearson_corr": 0.67,
                "test_rmse": 1.9,
                "selected_loss": "huber",
            },
            {
                "rank": 2,
                "run_label": "lstm",
                "model_name": "lstm",
                "model_group": "recurrent",
                "task": "regression",
                "lookback_steps": 48,
                "ranking_metric": "pearson_corr",
                "ranking_metric_value": 0.62,
                "test_pearson_corr": 0.62,
                "test_rmse": 2.2,
                "selected_loss": "huber",
            },
        ]
    )
    comparison_rows.to_parquet(dl_comparison_test_path, index=False)
    comparison_rows.to_parquet(dl_comparison_summary_path, index=False)
    pd.DataFrame(
        [
            {
                "strategy_name": "ridge_regression",
                "source": "baseline",
                "source_subtype": "linear",
                "task": "regression",
                "evaluation_split": "test",
                "has_trades": True,
                "trade_count": 4,
                "cumulative_return": 0.012,
                "annualized_return": 0.03,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.01,
                "mark_to_market_max_drawdown": -0.01,
                "realized_max_drawdown": -0.005,
                "win_rate": 0.5,
                "average_trade_return_bps": 6.0,
                "total_net_pnl_usd": 24.5,
                "final_equity_usd": 10024.5,
            }
        ]
    ).to_parquet(backtest_leaderboard_path, index=False)

    config = {
        "demo": {
            "title": "Demo",
            "subtitle": "Comparison-aware dashboard",
            "artifact_dir": str(artifact_dir),
            "frontend_public_dir": str(public_dir),
            "top_strategies": 3,
        },
        "contract": {
            "chain_name": "local_foundry",
            "asset_symbol": "mUSDC",
            "asset_decimals": 6,
            "demo_wallet_cash_assets": 1_000_000_000,
        },
        "inputs": {
            "data_manifest_path": str(data_manifest_path),
            "data_quality_summary_path": str(data_quality_summary_path),
            "backtest_manifest_path": str(backtest_manifest_path),
            "baseline_leaderboard_path": str(baseline_leaderboard_path),
            "dl_leaderboard_path": str(dl_leaderboard_path),
            "dl_comparison_manifest_path": str(dl_comparison_manifest_path),
            "dl_comparison_summary_path": str(dl_comparison_summary_path),
            "dl_comparison_test_leaderboard_path": str(dl_comparison_test_path),
            "backtest_leaderboard_path": str(backtest_leaderboard_path),
            "integration_selection_path": str(input_dir / "missing_selection.json"),
            "integration_plan_path": str(input_dir / "missing_plan.json"),
            "integration_call_summary_path": str(input_dir / "missing_calls.json"),
            "charts": [],
        },
    }

    try:
        artifacts = export_demo_snapshot(config)
        snapshot_payload = json.loads(
            Path(artifacts.artifact_snapshot_path).read_text(encoding="utf-8")
        )

        assert (
            snapshot_payload["models"]["deep_learning_best"]["model_name"]
            == "transformer_encoder"
        )
        assert snapshot_payload["models"]["deep_learning_comparison"]["available"] is True
        assert snapshot_payload["models"]["deep_learning_comparison"]["run_count"] == 4
        assert (
            len(snapshot_payload["models"]["deep_learning_comparison"]["test_leaderboard"])
            == 2
        )
        assert (
            snapshot_payload["models"]["deep_learning_single_best"]["model_name"]
            == "lstm"
        )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_export_demo_snapshot_loads_exploratory_showcase_json_inputs() -> None:
    tmp_path = _make_temp_dir()
    artifact_dir = tmp_path / "artifacts"
    public_dir = tmp_path / "public"
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    data_manifest_path = input_dir / "manifest.json"
    data_quality_summary_path = input_dir / "summary.json"
    backtest_manifest_path = input_dir / "backtest_manifest.json"
    baseline_leaderboard_path = input_dir / "baseline_leaderboard.parquet"
    backtest_leaderboard_path = input_dir / "backtest_leaderboard.parquet"
    exploratory_summary_path = input_dir / "exploratory_summary.json"
    exploratory_leaderboard_path = input_dir / "exploratory_leaderboard.json"
    exploratory_distribution_path = input_dir / "exploratory_distribution.json"
    exploratory_quantile_path = input_dir / "exploratory_quantile.json"

    _write_json(
        data_manifest_path,
        {
            "dataset": {"symbol": "BTCUSDT", "venue": "binance", "frequency": "1h"},
            "time_range": {
                "start": "2021-01-01T00:00:00+00:00",
                "end_exclusive": "2026-04-07T00:00:00+00:00",
            },
            "canonical_row_count": 100,
            "row_counts": {"perpetual_bars": 100},
        },
    )
    _write_json(
        data_quality_summary_path,
        {
            "funding_event_count": 12,
            "coverage": {"coverage_ratio": 1.0},
            "funding_mean_bps": 0.8,
            "funding_std_bps": 1.4,
            "spread_mean_bps": 2.1,
            "mean_perp_annualized_vol": 0.52,
        },
    )
    _write_json(
        backtest_manifest_path,
        {
            "summary": {
                "strategy_count": 1,
                "trade_count": 4,
                "primary_split": "test",
                "primary_trade_count": 4,
                "combined_trade_count": 4,
                "best_strategy": "ridge_regression",
            },
            "diagnostics": {"leverage": {}, "funding": {}},
            "assumptions": ["Prototype assumption one"],
        },
    )
    pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "ridge_regression",
                "task": "regression",
                "pearson_corr": 0.61,
                "rmse": 2.4,
            }
        ]
    ).to_parquet(baseline_leaderboard_path, index=False)
    pd.DataFrame(
        [
            {
                "strategy_name": "ridge_regression",
                "source": "baseline",
                "source_subtype": "linear",
                "task": "regression",
                "evaluation_split": "test",
                "has_trades": True,
                "trade_count": 4,
                "cumulative_return": 0.012,
                "annualized_return": 0.03,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.01,
                "mark_to_market_max_drawdown": -0.01,
                "realized_max_drawdown": -0.005,
                "win_rate": 0.5,
                "average_trade_return_bps": 6.0,
                "total_net_pnl_usd": 24.5,
                "final_equity_usd": 10024.5,
            }
        ]
    ).to_parquet(backtest_leaderboard_path, index=False)
    _write_json(
        exploratory_summary_path,
        {
            "disclaimer": "Exploratory results are supplementary.",
            "exploratory_summary": {
                "nonzero_trade_strategy_count": 2,
                "best_showcase_row": {
                    "strategy_name": "lstm_gross__rolling_top_decile_abs",
                    "trade_count": 12,
                },
            },
        },
    )
    _write_json(
        exploratory_leaderboard_path,
        [
            {
                "strategy_name": "lstm_gross__rolling_top_decile_abs",
                "model_name": "lstm",
                "target_type": "gross_opportunity_regression",
                "trade_count": 12,
                "status": "completed",
            }
        ],
    )
    _write_json(
        exploratory_distribution_path,
        [
            {"run_name": "lstm_gross", "split": "test", "signed_score_mean": 1.1},
        ],
    )
    _write_json(
        exploratory_quantile_path,
        [
            {
                "run_name": "lstm_gross",
                "absolute_score_quantile": 9,
                "avg_directional_return_bps": 2.3,
            }
        ],
    )

    config = {
        "demo": {
            "title": "Demo",
            "subtitle": "Exploratory-aware dashboard",
            "artifact_dir": str(artifact_dir),
            "frontend_public_dir": str(public_dir),
            "top_strategies": 3,
        },
        "contract": {
            "chain_name": "local_foundry",
            "asset_symbol": "mUSDC",
            "asset_decimals": 6,
            "demo_wallet_cash_assets": 1_000_000_000,
        },
        "inputs": {
            "data_manifest_path": str(data_manifest_path),
            "data_quality_summary_path": str(data_quality_summary_path),
            "backtest_manifest_path": str(backtest_manifest_path),
            "baseline_leaderboard_path": str(baseline_leaderboard_path),
            "dl_leaderboard_path": str(input_dir / "missing_dl.parquet"),
            "backtest_leaderboard_path": str(backtest_leaderboard_path),
            "integration_selection_path": str(input_dir / "missing_selection.json"),
            "integration_plan_path": str(input_dir / "missing_plan.json"),
            "integration_call_summary_path": str(input_dir / "missing_calls.json"),
            "exploratory_summary_path": str(exploratory_summary_path),
            "exploratory_leaderboard_path": str(exploratory_leaderboard_path),
            "exploratory_prediction_distribution_path": str(exploratory_distribution_path),
            "exploratory_quantile_analysis_path": str(exploratory_quantile_path),
            "charts": [],
        },
    }

    try:
        artifacts = export_demo_snapshot(config)
        snapshot_payload = json.loads(
            Path(artifacts.artifact_snapshot_path).read_text(encoding="utf-8")
        )

        assert snapshot_payload["exploratory_dl"]["available"] is True
        assert snapshot_payload["exploratory_dl"]["summary"]["best_showcase_row"]["strategy_name"] == "lstm_gross__rolling_top_decile_abs"
        assert len(snapshot_payload["exploratory_dl"]["leaderboard_preview"]) == 1
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
