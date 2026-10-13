from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from funding_arb.config.models import (
    ExploratoryDLDatasetSettings,
    ExploratoryDLReportSettings,
    ExploratoryDLSignalSettings,
)
from funding_arb.exploratory_dl.dataset import run_exploratory_dataset_pipeline
from funding_arb.exploratory_dl.reporting import run_exploratory_dl_report
from funding_arb.exploratory_dl.signals import run_exploratory_signal_generation

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_TMP_ROOT = REPO_ROOT / "tests" / ".tmp"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _make_temp_dir(name: str) -> Path:
    path = TEST_TMP_ROOT / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prediction_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    splits = ["train"] * 16 + ["validation"] * 16 + ["test"] * 16
    predicted = (
        [-2.8, -2.2, -1.9, -1.6, -1.2, -0.9, -0.6, -0.3, 0.3, 0.6, 0.9, 1.2, 1.6, 1.9, 2.2, 2.8]
        * 3
    )
    actual = [value * 1.4 for value in predicted]
    probability = [0.5 + max(min(value / 8.0, 0.35), -0.35) for value in predicted]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "split": splits,
            "model_name": ["lstm"] * len(timestamps),
            "model_family": ["deep_learning"] * len(timestamps),
            "task": ["regression"] * len(timestamps),
            "signal_direction": [
                "short_perp_long_spot" if value >= 0 else "long_perp_short_spot"
                for value in predicted
            ],
            "signal": [int(abs(value) >= 1.0) for value in predicted],
            "decision_score": predicted,
            "signal_threshold": [0.0] * len(timestamps),
            "signal_strength": [abs(value) for value in predicted],
            "predicted_probability": probability,
            "predicted_return_bps": predicted,
            "predicted_label": [int(value >= 0.0) for value in predicted],
            "actual_return_bps": actual,
            "actual_label": [int(value >= 0.0) for value in actual],
            "selected_hyperparameters_json": ["{}"] * len(timestamps),
            "prediction_mode": ["static"] * len(timestamps),
            "calibration_method": ["none"] * len(timestamps),
            "feature_importance_method": ["not_applicable"] * len(timestamps),
            "checkpoint_selection_metric": ["validation_avg_signal_return_bps"] * len(timestamps),
            "best_checkpoint_metric_value": [1.2] * len(timestamps),
            "checkpoint_selection_effective_metric": ["validation_avg_signal_return_bps"] * len(timestamps),
            "best_checkpoint_effective_metric_value": [1.2] * len(timestamps),
            "checkpoint_selection_fallback_used": [False] * len(timestamps),
            "selected_loss": ["huber"] * len(timestamps),
            "regression_loss": ["huber"] * len(timestamps),
            "use_balanced_classification_loss": [False] * len(timestamps),
            "preprocessing_scaler": ["robust"] * len(timestamps),
            "winsorize_lower_quantile": [0.01] * len(timestamps),
            "winsorize_upper_quantile": [0.99] * len(timestamps),
        }
    )


def test_exploratory_dataset_pipeline_builds_independent_targets() -> None:
    tmp_path = _make_temp_dir("exploratory-dataset")
    input_path = tmp_path / "strict_supervised.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "split": ["train", "validation", "test", "test"],
            "supervised_ready": [True, True, True, False],
            "target_future_gross_return_bps_24h": [5.0, -3.0, 0.0, 4.0],
            "feature_alpha": [1.0, 2.0, 3.0, 4.0],
        }
    ).to_parquet(input_path, index=False)

    settings = ExploratoryDLDatasetSettings.model_validate(
        {
            "input": {
                "source_dataset_path": str(input_path),
                "provider": "binance",
                "symbol": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
            },
            "output": {
                "output_dir": str(tmp_path / "processed"),
                "artifact_name": "dataset.parquet",
                "manifest_name": "manifest.json",
                "write_csv": False,
            },
        }
    )

    try:
        artifacts = run_exploratory_dataset_pipeline(settings)
        derived = pd.read_parquet(artifacts.dataset_path)
        manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))

        assert "target_future_short_perp_long_spot_gross_return_bps_24h" in derived.columns
        assert "target_future_signed_opportunity_bps_24h" in derived.columns
        assert "target_best_direction_is_short_perp_long_spot_24h" in derived.columns
        assert derived.loc[0, "target_best_direction_label_24h"] == "short_perp_long_spot"
        assert derived.loc[1, "target_best_direction_label_24h"] == "long_perp_short_spot"
        assert bool(derived.loc[0, "exploratory_ready"]) is True
        assert bool(derived.loc[3, "exploratory_ready"]) is False
        assert manifest["definitions"]["gross_opportunity_target"] == "target_future_short_perp_long_spot_gross_return_bps_24h"
        assert Path(artifacts.dataset_path) != input_path
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_exploratory_signal_generation_produces_nonzero_showcase_support() -> None:
    tmp_path = _make_temp_dir("exploratory-signals")
    prediction_path = tmp_path / "predictions.parquet"
    _prediction_frame().to_parquet(prediction_path, index=False)

    settings = ExploratoryDLSignalSettings.model_validate(
        {
            "input": {
                "provider": "binance",
                "symbol": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
                "runs": [
                    {
                        "name": "lstm_gross",
                        "prediction_path": str(prediction_path),
                        "target_type": "gross_opportunity_regression",
                        "task": "regression",
                    }
                ],
            },
            "ranking_rule": {
                "enabled": True,
                "name": "rolling_top_quartile_abs",
                "percentile_threshold": 0.75,
                "window_size": 8,
                "min_history": 6,
            },
            "threshold_rule": {
                "enabled": True,
                "name": "validation_tuned_support",
                "candidate_quantiles": [0.5, 0.75, 0.9],
                "min_signal_count": 3,
                "min_signal_rate": 0.1,
                "support_weight": 0.5,
            },
            "output": {
                "output_dir": str(tmp_path / "signals"),
                "artifact_name": "signals.parquet",
                "manifest_name": "signals_manifest.json",
                "strategy_catalog_name": "strategy_catalog.csv",
                "diagnostics_dir_name": "diagnostics",
                "write_csv": True,
            },
        }
    )

    try:
        artifacts = run_exploratory_signal_generation(settings)
        manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))
        signals = pd.read_parquet(artifacts.signals_path)
        strategy_catalog = pd.read_csv(artifacts.strategy_catalog_path)

        assert manifest["summary"]["active_signal_count"] > 0
        assert strategy_catalog["signal_rule_type"].tolist() == [
            "ranking_based",
            "threshold_based",
        ]
        assert signals["should_trade"].sum() > 0
        assert manifest["summary"]["signal_count_by_split"]["test"] > 0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_exploratory_report_writes_summary_and_frontend_exports() -> None:
    tmp_path = _make_temp_dir("exploratory-report")
    inputs_dir = tmp_path / "inputs"
    public_dir = tmp_path / "public"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    strict_snapshot_path = inputs_dir / "strict_demo_snapshot.json"
    strict_final_summary_path = inputs_dir / "strict_final_summary.json"
    strict_comparison_path = inputs_dir / "strict_comparison.parquet"
    exploratory_comparison_path = inputs_dir / "exploratory_comparison.parquet"
    prediction_path = inputs_dir / "predictions.parquet"
    signal_manifest_path = inputs_dir / "signals_manifest.json"
    strategy_catalog_path = inputs_dir / "strategy_catalog.csv"
    signals_path = inputs_dir / "signals.parquet"
    backtest_manifest_path = inputs_dir / "backtest_manifest.json"
    backtest_leaderboard_path = inputs_dir / "leaderboard.parquet"
    trade_log_path = inputs_dir / "trade_log.parquet"
    dataset_manifest_path = inputs_dir / "dataset_manifest.json"

    _prediction_frame().to_parquet(prediction_path, index=False)
    strict_snapshot_path.write_text(
        json.dumps(
            {
                "meta": {"venue": "binance", "symbol": "BTCUSDT", "frequency": "1h"},
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
                    "summary": {"best_strategy_status": "unprofitable"},
                    "best_strategy": {
                        "strategy_name": "spread_zscore_1p5",
                        "trade_count": 200,
                    },
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    strict_final_summary_path.write_text(json.dumps({"report": "strict"}), encoding="utf-8")
    dataset_manifest_path.write_text(json.dumps({"dataset": "exploratory"}), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "transformer_encoder",
                "task": "regression",
                "target_column": "target_future_net_return_bps_24h",
                "status": "ok",
                "pearson_corr": 0.646,
                "rmse": 1.2,
                "signal_count": 0,
            }
        ]
    ).to_parquet(strict_comparison_path, index=False)
    pd.DataFrame(
        [
            {
                "split": "test",
                "model_name": "lstm",
                "task": "regression",
                "target_column": "target_future_short_perp_long_spot_gross_return_bps_24h",
                "status": "ok",
                "pearson_corr": 0.51,
                "rmse": 2.1,
                "signal_count": 14,
            }
        ]
    ).to_parquet(exploratory_comparison_path, index=False)

    strategy_catalog = pd.DataFrame(
        [
            {
                "strategy_name": "lstm_gross__rolling_top_quartile_abs",
                "model_name": "lstm",
                "run_name": "lstm_gross",
                "target_type": "gross_opportunity_regression",
                "task": "regression",
                "signal_rule": "rolling_top_quartile_abs",
                "signal_rule_type": "ranking_based",
                "selection_reason": "rolling percentile",
                "status": "ok",
                "reason": None,
            }
        ]
    )
    strategy_catalog.to_csv(strategy_catalog_path, index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2024-01-02T00:00:00+00:00",
                "asset": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
                "source": "exploratory_dl",
                "source_subtype": "deep_learning_showcase",
                "strategy_name": "lstm_gross__rolling_top_quartile_abs",
                "model_family": "deep_learning",
                "task": "regression",
                "signal_score": 2.2,
                "predicted_class": 1,
                "expected_return_bps": 2.2,
                "signal_threshold": 1.8,
                "threshold_objective": "balanced_avg_return_support",
                "selected_threshold_objective_value": 1.2,
                "prediction_mode": "static",
                "calibration_method": "none",
                "feature_importance_method": "not_applicable",
                "selected_hyperparameters_json": "{}",
                "checkpoint_selection_metric": "validation_avg_signal_return_bps",
                "best_checkpoint_metric_value": 1.2,
                "checkpoint_selection_effective_metric": "validation_avg_signal_return_bps",
                "best_checkpoint_effective_metric_value": 1.2,
                "checkpoint_selection_fallback_used": False,
                "selected_loss": "huber",
                "regression_loss": "huber",
                "use_balanced_classification_loss": False,
                "preprocessing_scaler": "robust",
                "winsorize_lower_quantile": 0.01,
                "winsorize_upper_quantile": 0.99,
                "suggested_direction": "short_perp_long_spot",
                "confidence": 2.2,
                "should_trade": 1,
                "split": "test",
                "metadata_json": "{}",
            }
        ]
    ).to_parquet(signals_path, index=False)
    signal_manifest_path.write_text(
        json.dumps(
            {
                "input": {
                    "runs": [
                        {
                            "name": "lstm_gross",
                            "prediction_path": str(prediction_path),
                            "target_type": "gross_opportunity_regression",
                            "task": "regression",
                            "enabled": True,
                        }
                    ]
                },
                "strategy_catalog_path": str(strategy_catalog_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    backtest_manifest_path.write_text(
        json.dumps({"summary": {"strategy_count": 1}}, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "strategy_name": "lstm_gross__rolling_top_quartile_abs",
                "source": "exploratory_dl",
                "source_subtype": "deep_learning_showcase",
                "task": "regression",
                "evaluation_split": "test",
                "status": "completed",
                "diagnostic_reason": None,
                "trade_count": 5,
                "cumulative_return": 0.012,
                "mark_to_market_max_drawdown": -0.01,
                "sharpe_ratio": 1.4,
                "total_net_pnl_usd": 120.0,
            }
        ]
    ).to_parquet(backtest_leaderboard_path, index=False)
    pd.DataFrame(
        [
            {
                "strategy_name": "lstm_gross__rolling_top_quartile_abs",
                "direction": "short_perp_long_spot",
                "net_pnl_usd": 120.0,
                "net_return_bps": 12.0,
            }
        ]
    ).to_parquet(trade_log_path, index=False)

    settings = ExploratoryDLReportSettings.model_validate(
        {
            "input": {
                "strict_demo_snapshot_path": str(strict_snapshot_path),
                "strict_final_report_summary_path": str(strict_final_summary_path),
                "strict_comparison_summary_path": str(strict_comparison_path),
                "exploratory_dataset_manifest_path": str(dataset_manifest_path),
                "exploratory_comparison_summary_path": str(exploratory_comparison_path),
                "exploratory_signals_path": str(signals_path),
                "exploratory_signals_manifest_path": str(signal_manifest_path),
                "exploratory_backtest_manifest_path": str(backtest_manifest_path),
                "exploratory_backtest_leaderboard_path": str(backtest_leaderboard_path),
                "exploratory_trade_log_path": str(trade_log_path),
            },
            "output": {
                "output_dir": str(tmp_path / "reports"),
                "frontend_public_dir": str(public_dir),
                "write_csv": True,
                "write_markdown": True,
                "write_json_summary": True,
                "write_plots": True,
                "copy_to_frontend_public": True,
            },
        }
    )

    try:
        artifacts = run_exploratory_dl_report(settings)
        summary_payload = json.loads(Path(artifacts.summary_json_path or "").read_text(encoding="utf-8"))

        assert summary_payload["exploratory_summary"]["nonzero_trade_strategy_count"] == 1
        assert len(summary_payload["exploratory_summary"]["figure_assets"]) >= 1
        assert Path(artifacts.full_leaderboard_path).exists()
        assert (public_dir / "exploratory_dl_summary.json").exists()
        assert (public_dir / "exploratory_dl_leaderboard.json").exists()
        assert (public_dir / "exploratory_prediction_distribution.json").exists()
        assert (public_dir / "exploratory_quantile_analysis.json").exists()
        assert any((public_dir / "assets").glob("*.png"))
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
