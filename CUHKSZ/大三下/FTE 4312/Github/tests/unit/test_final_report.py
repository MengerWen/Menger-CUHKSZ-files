from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from funding_arb.config.models import FinalReportSettings
from funding_arb.reporting.final_report import run_final_report

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = REPO_ROOT / "tests" / ".tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_temp_dir() -> Path:
    path = TEST_TMP_ROOT / f"final-report-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_generate_final_report_writes_artifact_and_public_outputs() -> None:
    tmp_path = _make_temp_dir()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = tmp_path / "reports"
    public_dir = tmp_path / "public"
    chart_path = input_dir / "chart.png"
    chart_path.write_bytes(b"fake-chart")

    snapshot_path = input_dir / "demo_snapshot.json"
    robustness_path = input_dir / "robustness_summary.json"

    _write_json(
        snapshot_path,
        {
            "meta": {
                "date_range": {
                    "start": "2021-01-01T00:00:00+00:00",
                    "end_exclusive": "2026-04-08T00:00:00+00:00",
                }
            },
            "research": {
                "canonical_rows": 46152,
                "funding_events": 3092,
                "coverage_ratio": 1.0,
                "funding_mean_bps": 1.04,
                "funding_std_bps": 1.89,
                "spread_mean_bps": -1.53,
                "annualized_volatility": 0.51,
            },
            "models": {
                "baseline_best": {
                    "model_name": "elastic_net_regression",
                    "pearson_corr": 0.677,
                    "rmse": 1.269,
                    "signal_count": 0,
                },
                "deep_learning_best": {
                    "model_name": "transformer_encoder",
                    "ranking_metric": "pearson_corr",
                    "ranking_metric_value": 0.646,
                    "test_rmse": 1.209,
                    "test_signal_count": 0,
                },
            },
            "backtest": {
                "summary": {"primary_split": "test"},
                "best_strategy": {
                    "strategy_name": "spread_zscore_1p5",
                    "trade_count": 200,
                    "cumulative_return": -0.0647,
                    "sharpe_ratio": -14.07,
                    "total_net_pnl_usd": -6474.85,
                },
                "top_strategies": [
                    {
                        "strategy_name": "spread_zscore_1p5",
                        "source_subtype": "rule_based",
                        "evaluation_split": "test",
                        "has_trades": True,
                        "trade_count": 200,
                        "cumulative_return": -0.0647,
                        "mark_to_market_max_drawdown": -0.0647,
                        "sharpe_ratio": -14.07,
                        "total_net_pnl_usd": -6474.85,
                    }
                ],
                "assumptions": ["Prototype assumption one", "Prototype assumption two"],
            },
            "vault": {
                "selected_strategy": "spread_zscore_1p5",
                "strategy_state": "idle",
                "suggested_direction": "flat",
                "reported_nav_assets": 93525146215,
                "summary_pnl_usd": -6474.85,
                "call_count": 2,
            },
            "charts": [
                {
                    "title": "Funding Regime",
                    "subtitle": "Synthetic chart",
                    "section": "data",
                    "source_path": str(chart_path),
                }
            ],
        },
    )
    _write_json(
        robustness_path,
        {
            "ranking_metric": "cumulative_return",
            "family_comparison": [
                {
                    "family_label": "Deep Learning",
                    "strategy_name": "lstm",
                    "trade_count": 0,
                    "cumulative_return": 0.0,
                    "sharpe_ratio": 0.0,
                    "total_net_pnl_usd": 0.0,
                }
            ],
        },
    )

    config = FinalReportSettings.model_validate(
        {
            "metadata": {
                "title": "Project Final Report",
                "subtitle": "A full test report",
                "course": "FTE 4312",
                "authors": ["A", "B"],
                "repository_url": "https://example.com/repo",
                "provider": "binance",
                "symbol": "BTCUSDT",
                "frequency": "1h",
            },
            "input": {
                "demo_snapshot_path": str(snapshot_path),
                "robustness_summary_path": str(robustness_path),
            },
            "sections": {
                "executive_summary": ["Summary line one"],
                "contributions": ["Contribution one"],
                "limitations": ["Limitation one"],
                "future_work": ["Future work one"],
            },
            "output": {
                "artifact_dir": str(artifact_dir),
                "frontend_public_dir": str(public_dir),
                "write_markdown": True,
                "write_html": True,
                "write_json_summary": True,
                "copy_to_frontend_public": True,
            },
        }
    )

    try:
        artifacts = run_final_report(config)
        artifact_root = Path(artifacts.artifact_output_dir)
        public_root = Path(artifacts.public_report_dir or "")

        assert (artifact_root / "final_report.md").exists()
        assert (artifact_root / "final_report.html").exists()
        assert (artifact_root / "summary.json").exists()
        assert (artifact_root / "assets" / "chart.png").exists()

        assert (public_root / "index.html").exists()
        assert (public_root / "final_report.md").exists()
        assert (public_root / "summary.json").exists()
        assert (public_root / "assets" / "chart.png").exists()

        markdown_text = (artifact_root / "final_report.md").read_text(encoding="utf-8")
        summary_payload = json.loads((artifact_root / "summary.json").read_text(encoding="utf-8"))

        assert "Project Final Report" in markdown_text
        assert "spread_zscore_1p5" in markdown_text
        assert summary_payload["meta"]["title"] == "Project Final Report"
        assert summary_payload["best_family_note"] is not None
        assert summary_payload["charts"][0]["artifact_path"] == "assets/chart.png"
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_generate_final_report_can_embed_exploratory_summary() -> None:
    tmp_path = _make_temp_dir()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = tmp_path / "reports"
    public_dir = tmp_path / "public"

    snapshot_path = input_dir / "demo_snapshot.json"
    robustness_path = input_dir / "robustness_summary.json"
    exploratory_summary_path = input_dir / "exploratory_summary.json"

    _write_json(
        snapshot_path,
        {
            "meta": {
                "date_range": {
                    "start": "2021-01-01T00:00:00+00:00",
                    "end_exclusive": "2026-04-08T00:00:00+00:00",
                }
            },
            "research": {
                "canonical_rows": 46152,
                "funding_events": 3092,
                "coverage_ratio": 1.0,
                "funding_mean_bps": 1.04,
                "funding_std_bps": 1.89,
                "spread_mean_bps": -1.53,
                "annualized_volatility": 0.51,
            },
            "models": {
                "baseline_best": {
                    "model_name": "elastic_net_regression",
                    "pearson_corr": 0.677,
                    "signal_count": 0,
                },
                "deep_learning_best": {
                    "model_name": "transformer_encoder",
                    "pearson_corr": 0.646,
                    "signal_count": 0,
                },
            },
            "backtest": {
                "summary": {"primary_split": "test"},
                "best_strategy": {
                    "strategy_name": "spread_zscore_1p5",
                    "trade_count": 200,
                    "cumulative_return": -0.0647,
                    "sharpe_ratio": -14.07,
                    "total_net_pnl_usd": -6474.85,
                },
                "top_strategies": [],
                "assumptions": [],
            },
            "vault": {
                "selected_strategy": "spread_zscore_1p5",
                "strategy_state": "idle",
                "suggested_direction": "flat",
                "reported_nav_assets": 93525146215,
                "summary_pnl_usd": -6474.85,
                "call_count": 2,
            },
            "charts": [],
        },
    )
    _write_json(robustness_path, {"family_comparison": []})
    _write_json(
        exploratory_summary_path,
        {
            "disclaimer": "Exploratory results are supplementary only.",
            "exploratory_summary": {
                "nonzero_trade_strategy_count": 3,
                "best_showcase_row": {
                    "strategy_name": "transformer_direction__rolling_top_decile_abs",
                    "trade_count": 12,
                    "cumulative_return": 0.018,
                    "total_net_pnl_usd": 180.0,
                },
            },
        },
    )

    config = FinalReportSettings.model_validate(
        {
            "input": {
                "demo_snapshot_path": str(snapshot_path),
                "robustness_summary_path": str(robustness_path),
                "exploratory_summary_path": str(exploratory_summary_path),
            },
            "output": {
                "artifact_dir": str(artifact_dir),
                "frontend_public_dir": str(public_dir),
                "write_markdown": True,
                "write_html": False,
                "write_json_summary": True,
                "copy_to_frontend_public": False,
            },
        }
    )

    try:
        artifacts = run_final_report(config)
        markdown_text = (
            Path(artifacts.artifact_output_dir) / "final_report.md"
        ).read_text(encoding="utf-8")

        assert "Exploratory DL Showcase" in markdown_text
        assert "supplementary" in markdown_text
        assert "transformer_direction__rolling_top_decile_abs" in markdown_text
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
