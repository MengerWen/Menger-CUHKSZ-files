from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from funding_arb.demo_showcase import DemoShowcaseSettings, run_demo_showcase

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = REPO_ROOT / "tests" / ".tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _make_temp_dir() -> Path:
    path = TEST_TMP_ROOT / f"demo-showcase-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_run_demo_showcase_builds_isolated_showcase_bundle() -> None:
    tmp_path = _make_temp_dir()
    try:
        config = DemoShowcaseSettings.model_validate(
            {
                "metadata": {
                    "title": "Presentation Showcase",
                    "subtitle": "Showcase bundle",
                    "provider": "binance",
                    "symbol": "BTCUSDT",
                    "venue": "binance",
                    "frequency": "1h",
                    "frontend_bundle_name": "demo_showcase",
                },
                "paths": {
                    "data_dir": str(tmp_path / "data_demo"),
                    "reports_dir": str(tmp_path / "reports_demo"),
                    "frontend_public_dir": str(tmp_path / "public_demo"),
                },
                "generation": {
                    "seed": 7,
                    "data_start": "2024-01-01T00:00:00+00:00",
                    "data_end": "2024-08-01T00:00:00+00:00",
                    "train_start": "2024-02-01T00:00:00+00:00",
                    "validation_start": "2024-04-01T00:00:00+00:00",
                    "test_start": "2024-05-01T00:00:00+00:00",
                    "test_end": "2024-07-31T00:00:00+00:00",
                    "initial_capital_usd": 100000.0,
                    "position_notional_usd": 20000.0,
                },
                "sections": {
                    "executive_summary": ["Showcase summary."],
                    "contributions": ["Presentation artifact bundle."],
                    "limitations": ["Presentation-focused output."],
                    "future_work": ["Add more preset packs."],
                },
                "strict_strategies": [
                    {
                        "strategy_name": "elastic_net_regression",
                        "display_name": "Baseline ML",
                        "family_label": "Simple ML Baseline",
                        "track": "strict",
                        "source": "baseline",
                        "source_subtype": "baseline_linear",
                        "task": "regression",
                        "model_name": "elastic_net_regression",
                        "model_group": "linear",
                        "style": "baseline_ml",
                        "target_cumulative_return": 0.038,
                        "target_sharpe": 0.72,
                        "target_max_drawdown": -0.041,
                        "trade_count": 42,
                        "signal_count_test": 58,
                        "signal_count_validation": 44,
                        "signal_count_train": 101,
                        "model_metric_value": 0.55,
                        "rmse": 2.1,
                        "win_rate": 0.54,
                        "story_note": "Baseline test note.",
                    },
                    {
                        "strategy_name": "transformer_encoder",
                        "display_name": "TransformerEncoder",
                        "family_label": "Deep Learning",
                        "track": "strict",
                        "source": "dl",
                        "source_subtype": "deep_learning",
                        "task": "regression",
                        "model_name": "transformer_encoder",
                        "model_group": "attention",
                        "style": "transformer_high_conviction",
                        "target_cumulative_return": 0.095,
                        "target_sharpe": 1.36,
                        "target_max_drawdown": -0.049,
                        "trade_count": 38,
                        "signal_count_test": 51,
                        "signal_count_validation": 39,
                        "signal_count_train": 96,
                        "model_metric_value": 0.67,
                        "rmse": 1.66,
                        "win_rate": 0.6,
                        "ranking_metric_value": 0.67,
                        "story_note": "Transformer test note.",
                    },
                ],
                "exploratory_strategies": [
                    {
                        "strategy_name": "transformer_direction__rolling_top_decile_abs",
                        "display_name": "Exploratory Transformer Direction",
                        "family_label": "Exploratory Deep Learning",
                        "track": "exploratory",
                        "source": "exploratory_dl",
                        "source_subtype": "deep_learning_showcase",
                        "task": "classification",
                        "model_name": "transformer_encoder",
                        "model_group": "attention",
                        "style": "exploratory_directional",
                        "target_cumulative_return": 0.142,
                        "target_sharpe": 1.48,
                        "target_max_drawdown": -0.091,
                        "trade_count": 74,
                        "signal_count_test": 95,
                        "signal_count_validation": 66,
                        "signal_count_train": 149,
                        "model_metric_value": 0.69,
                        "rmse": 1.61,
                        "win_rate": 0.57,
                        "target_type": "direction_classification",
                        "signal_rule": "rolling_top_decile_abs",
                        "signal_rule_type": "ranking_based",
                        "story_note": "Exploratory test note.",
                    }
                ],
            }
        )

        artifacts = run_demo_showcase(config)

        snapshot = json.loads(Path(artifacts.snapshot_path).read_text(encoding="utf-8"))
        assert snapshot["meta"]["artifact_label"] == "SHOWCASE"
        assert Path(artifacts.frontend_snapshot_path).exists()
        assert Path(artifacts.final_report_path or "").exists()
        assert Path(artifacts.final_report_html_path or "").exists()
        assert Path(artifacts.final_report_summary_path or "").exists()
        assert Path(artifacts.modeling_summary_path).exists()
        assert Path(artifacts.backtest_summary_path).exists()
        assert Path(artifacts.exploratory_summary_path).exists()

        strict_leaderboard = pd.read_parquet(
            Path(artifacts.data_root) / "backtests" / "strict" / "leaderboard.parquet"
        )
        assert "artifact_label" in strict_leaderboard.columns
        assert strict_leaderboard["artifact_label"].eq("SHOWCASE").all()

        final_report_markdown = Path(artifacts.final_report_path or "").read_text(
            encoding="utf-8"
        )
        assert "SHOWCASE" in final_report_markdown

        frontend_report_index = Path(artifacts.frontend_public_dir) / "report" / "index.html"
        assert frontend_report_index.exists()
        assert Path(artifacts.manifest_path).exists()
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
