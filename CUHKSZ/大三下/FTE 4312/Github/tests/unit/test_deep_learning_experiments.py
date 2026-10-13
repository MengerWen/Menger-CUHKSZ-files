from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from funding_arb.config.models import (
    DeepLearningComparisonRunSettings,
    DeepLearningComparisonSettings,
    DeepLearningSettings,
)
from funding_arb.models.deep_learning_experiments import (
    _resolve_run_settings,
    run_deep_learning_comparison,
)


SCRATCH_ROOT = Path("tests/.tmp/deep_learning_experiments")


def _base_dl_config(
    model_dir: Path,
    run_name: str,
    *,
    model_name: str,
) -> dict[str, Any]:
    return DeepLearningSettings.model_validate(
        {
            "input": {
                "dataset_path": "data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_dataset.parquet",
                "provider": "binance",
                "symbol": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
            },
            "target": {
                "task": "regression",
                "column": "target_future_net_return_bps_24h",
                "classification_column": "target_is_profitable_24h",
                "regression_column": "target_future_net_return_bps_24h",
                "timestamp_column": "timestamp",
                "split_column": "split",
                "ready_column": "supervised_ready",
            },
            "feature_selection": {},
            "sequence": {"lookback_steps": 4, "allow_cross_split_context": True},
            "model": {
                "name": model_name,
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
                "transformer_d_model": 8,
                "transformer_nhead": 2,
                "transformer_num_layers": 1,
                "transformer_dim_feedforward": 16,
            },
            "training": {
                "batch_size": 4,
                "epochs": 1,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "seed": 42,
                "device": "cpu",
                "num_workers": 0,
                "clip_grad_norm": 1.0,
                "early_stopping_patience": 1,
                "deterministic": True,
            },
            "output": {
                "model_dir": str(model_dir),
                "run_name": run_name,
                "write_csv": True,
                "write_markdown_report": True,
            },
        }
    ).model_dump()


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _write_fake_artifacts(
    model_dir: Path,
    run_name: str,
    *,
    model_name: str,
    validation_score: float,
    test_score: float,
) -> None:
    output_dir = model_dir / "binance" / "btcusdt" / "1h" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard_path = output_dir / "dl_leaderboard.parquet"
    report_path = output_dir / "training_report.md"
    report_path.write_text("# Fake DL Report\n", encoding="utf-8")
    leaderboard = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "model_family": "deep_learning",
                "task": "regression",
                "split": "validation",
                "row_count": 10,
                "pearson_corr": validation_score,
                "rmse": 2.0,
                "avg_signal_return_bps": 1.0,
                "cumulative_signal_return_bps": 10.0,
                "signal_hit_rate": 0.6,
                "signal_count": 3,
                "top_quantile_avg_return_bps": validation_score * 10.0,
                "top_quantile_cumulative_return_bps": validation_score * 100.0,
            },
            {
                "model_name": model_name,
                "model_family": "deep_learning",
                "task": "regression",
                "split": "test",
                "row_count": 12,
                "pearson_corr": test_score,
                "rmse": 2.5,
                "avg_signal_return_bps": 2.0,
                "cumulative_signal_return_bps": 20.0,
                "signal_hit_rate": 0.7,
                "signal_count": 4,
                "top_quantile_avg_return_bps": test_score * 10.0,
                "top_quantile_cumulative_return_bps": test_score * 100.0,
            },
        ]
    )
    leaderboard.to_parquet(leaderboard_path, index=False)
    manifest = {
        "best_epoch": 2,
        "best_checkpoint_metric": "validation_pearson_corr",
        "best_checkpoint_metric_value": validation_score,
        "best_checkpoint_effective_metric": "validation_pearson_corr",
        "best_checkpoint_effective_metric_value": validation_score,
        "checkpoint_metric_fallback_used": False,
        "status": "ok",
        "reason": None,
        "degenerate_experiment": False,
        "degenerate_stage": None,
        "degenerate_reason": None,
        "fallback_used": False,
        "fallback_reason": None,
        "selected_threshold": 0.0,
        "selected_threshold_objective": "avg_signal_return_bps",
        "selected_threshold_objective_value": 1.0,
        "selected_loss": "huber",
        "prediction_mode": "static",
        "selected_hyperparameters": {
            "model_name": model_name,
            "lookback_steps": 4,
            "hidden_size": 8,
        },
        "checkpoint_path": str(output_dir / "best_model.pt"),
        "history_path": str(output_dir / "training_history.csv"),
        "predictions_path": str(output_dir / "dl_predictions.parquet"),
        "metrics_path": str(output_dir / "dl_metrics.parquet"),
        "leaderboard_path": str(leaderboard_path),
        "report_path": str(report_path),
    }
    (output_dir / "dl_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def test_resolve_run_settings_applies_nested_overrides() -> None:
    scratch = SCRATCH_ROOT / "resolve_overrides"
    shutil.rmtree(scratch, ignore_errors=True)
    config_path = scratch / "lstm.yaml"
    _write_config(
        config_path,
        _base_dl_config(scratch / "models", "lstm_regression_unit", model_name="lstm"),
    )

    _, settings = _resolve_run_settings(
        DeepLearningComparisonRunSettings.model_validate(
            {
                "name": "lstm_classification",
                "config_path": str(config_path.resolve()),
                "overrides": {
                    "target": {
                        "task": "classification",
                        "column": "target_is_profitable_24h",
                    },
                    "output": {"run_name": "lstm_classification_unit"},
                },
            }
        )
    )

    assert settings.model.name == "lstm"
    assert settings.target.task == "classification"
    assert settings.target.column == "target_is_profitable_24h"
    assert settings.output.run_name == "lstm_classification_unit"


def test_run_deep_learning_comparison_reuses_artifacts_and_writes_tables() -> None:
    scratch = SCRATCH_ROOT / "reuse_artifacts"
    shutil.rmtree(scratch, ignore_errors=True)
    model_dir = scratch / "models"
    comparison_dir = scratch / "comparisons"
    lstm_config = scratch / "configs" / "lstm.yaml"
    gru_config = scratch / "configs" / "gru.yaml"

    _write_config(lstm_config, _base_dl_config(model_dir, "lstm_unit", model_name="lstm"))
    _write_config(gru_config, _base_dl_config(model_dir, "gru_unit", model_name="gru"))
    _write_fake_artifacts(
        model_dir,
        "lstm_unit",
        model_name="lstm",
        validation_score=0.7,
        test_score=0.6,
    )
    _write_fake_artifacts(
        model_dir,
        "gru_unit",
        model_name="gru",
        validation_score=0.5,
        test_score=0.4,
    )

    settings = DeepLearningComparisonSettings.model_validate(
        {
            "experiment_name": "unit_sequence_regression",
            "description": "unit test bundle",
            "runner": {"train_if_missing": False, "fail_fast": True},
            "ranking": {
                "validation_metric": "pearson_corr",
                "test_metric": "pearson_corr",
                "strategy_metric": "top_quantile_avg_return_bps",
                "strategy_split": "test",
            },
            "runs": [
                {"name": "lstm", "config_path": str(lstm_config.resolve())},
                {"name": "gru", "config_path": str(gru_config.resolve())},
            ],
            "output": {
                "output_dir": str(comparison_dir),
                "run_name": "unit_bundle",
                "write_csv": True,
                "write_markdown_report": True,
                "write_plots": True,
            },
        }
    )

    artifacts = run_deep_learning_comparison(settings)

    summary = pd.read_parquet(artifacts.comparison_summary_path)
    validation = pd.read_parquet(artifacts.validation_leaderboard_path)
    test = pd.read_parquet(artifacts.test_leaderboard_path)
    manifest = json.loads(Path(artifacts.manifest_path).read_text(encoding="utf-8"))

    assert len(summary) == 2
    assert set(summary["model_name"]) == {"lstm", "gru"}
    assert summary["artifact_reused"].all()
    assert set(summary["status"]) == {"ok"}
    assert set(summary["degenerate_experiment"]) == {False}
    assert "manifest_path" in summary.columns
    assert "report_path" in summary.columns
    assert "status" in validation.columns
    assert validation.iloc[0]["run_label"] == "lstm"
    assert test.iloc[0]["run_label"] == "lstm"
    assert manifest["run_count"] == 2
    assert manifest["runs"][0]["status"] == "ok"
    assert Path(artifacts.report_path).exists()
    assert len(artifacts.figure_paths) >= 2
