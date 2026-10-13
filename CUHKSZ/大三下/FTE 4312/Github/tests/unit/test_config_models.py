from __future__ import annotations

from pathlib import Path

import pytest

from funding_arb.config.loader import get_command_settings, load_command_settings, load_settings
from funding_arb.config.models import (
    BacktestSettings,
    BaselineSettings,
    DataQualityReportSettings,
    DataSettings,
    DemoWorkflowSettings,
    DeepLearningComparisonSettings,
    DeepLearningSettings,
    ExploratoryDLDatasetSettings,
    ExploratoryDLReportSettings,
    ExploratoryDLSignalSettings,
    FeatureSettings,
    IntegrationSettings,
    LabelPipelineSettings,
    RobustnessReportSettings,
    SignalSettings,
)


def test_fetch_data_default_config_loads_typed_model() -> None:
    config = load_command_settings("fetch-data")
    assert isinstance(config, DataSettings)
    assert config.dataset.symbol == "BTCUSDT"
    assert config.source.provider == "binance"
    assert config.sources["funding"].enabled is True


def test_report_data_quality_default_config_loads_typed_model() -> None:
    config = load_command_settings("report-data-quality")
    assert isinstance(config, DataQualityReportSettings)
    assert config.input.symbol == "BTCUSDT"
    assert config.output.output_dir == "reports/data_quality"


def test_build_features_default_config_loads_typed_model() -> None:
    config = load_command_settings("build-features")
    assert isinstance(config, FeatureSettings)
    assert config.input.dataset_path.endswith("hourly_market_data.parquet")
    assert config.feature_set.rolling_windows == [8, 24, 72, 168]
    assert config.labels.forward_horizon_hours == 8


def test_build_labels_default_config_loads_typed_model() -> None:
    config = load_command_settings("build-labels")
    assert isinstance(config, LabelPipelineSettings)
    assert config.input.feature_table_path.endswith("btcusdt_feature_set.parquet")
    assert config.target.holding_windows_hours == [8, 24]
    assert config.target.primary_horizon_hours == 8


def test_train_baseline_default_config_loads_typed_model() -> None:
    config = load_command_settings("train-baseline")
    assert isinstance(config, BaselineSettings)
    assert config.input.dataset_path.endswith("btcusdt_supervised_dataset.parquet")
    assert config.target.classification_column == "target_is_profitable_24h"
    assert config.predictive.classification.enabled is True
    assert config.tuning.gap == 24
    assert config.threshold_search.objective == "avg_signal_return_bps"
    assert config.threshold_search.allow_degenerate_fallback is False
    assert config.imputation.add_missing_indicators is True
    assert len(config.predictive.classification.additional_models) >= 2
    assert config.predictive.classification.calibration_method == "sigmoid"
    assert len(config.rules) >= 1


def test_train_dl_default_config_loads_typed_model() -> None:
    config = load_command_settings("train-dl")
    assert isinstance(config, DeepLearningSettings)
    assert config.input.dataset_path.endswith("btcusdt_supervised_dataset.parquet")
    assert config.target.task == "regression"
    assert config.sequence.lookback_steps == 48
    assert config.model.name == "lstm"
    assert config.training.selection_metric == "validation_avg_signal_return_bps"
    assert config.threshold_search.enabled is True
    assert config.threshold_search.allow_degenerate_fallback is False
    assert config.training.allow_degenerate_fallback is False
    assert config.preprocessing.scaler == "robust"
    assert config.prediction.mode == "static"


def test_additional_dl_model_configs_load_typed_models() -> None:
    gru = load_settings(Path("configs/models/gru.yaml"), DeepLearningSettings)
    tcn = load_settings(Path("configs/models/tcn.yaml"), DeepLearningSettings)
    transformer = load_settings(Path("configs/models/transformer.yaml"), DeepLearningSettings)

    assert gru.model.name == "gru"
    assert gru.output.run_name == "gru_regression_24h_default"
    assert tcn.model.name == "tcn"
    assert tcn.model.tcn_num_blocks == 4
    assert transformer.model.name == "transformer_encoder"
    assert transformer.model.transformer_nhead == 4


def test_dl_comparison_bundle_configs_load_typed_models() -> None:
    regression = load_settings(
        Path("configs/experiments/dl/regression_all.yaml"),
        DeepLearningComparisonSettings,
    )
    recurrent = load_settings(
        Path("configs/experiments/dl/recurrent_regression.yaml"),
        DeepLearningComparisonSettings,
    )
    classification = load_settings(
        Path("configs/experiments/dl/classification_all.yaml"),
        DeepLearningComparisonSettings,
    )

    assert regression.experiment_name == "sequence_regression_all"
    assert len(regression.runs) == 4
    assert recurrent.experiment_name == "recurrent_regression_only"
    assert len(recurrent.runs) == 2
    assert classification.experiment_name == "sequence_classification_all"
    assert classification.ranking.test_metric == "f1"
    assert classification.runs[0].overrides["target"]["task"] == "classification"


def test_generate_signals_default_config_loads_typed_model() -> None:
    config = load_command_settings("generate-signals")
    assert isinstance(config, SignalSettings)
    assert config.input.baseline_predictions_path.endswith(
        "baseline_predictions.parquet"
    )
    assert "transformer_regression_24h_default" in config.input.dl_predictions_path
    assert config.input.dl_predictions_path.endswith("dl_predictions.parquet")
    assert config.source.name == "baseline"


def test_backtest_default_config_loads_typed_model() -> None:
    config = load_command_settings("backtest")
    assert isinstance(config, BacktestSettings)
    assert config.input.signal_path.endswith("signals.parquet")
    assert config.input.market_dataset_path.endswith("hourly_market_data.parquet")
    assert config.execution.holding_window_hours == 24
    assert config.reporting.run_name == "baseline_signals_default"


def test_backtest_primary_split_must_be_selected() -> None:
    config = load_command_settings("backtest").model_dump()
    config["selection"]["split_filter"] = ["train", "validation"]
    config["reporting"]["primary_split"] = "test"

    with pytest.raises(ValueError, match="primary_split"):
        BacktestSettings.model_validate(config)


def test_backtest_empty_split_filter_means_all_splits() -> None:
    config = load_command_settings("backtest").model_dump()
    config["selection"]["split_filter"] = []
    config["reporting"]["primary_split"] = "test"

    settings = BacktestSettings.model_validate(config)

    assert settings.selection.split_filter == []
    assert settings.reporting.primary_split == "test"


def test_robustness_report_default_config_loads_typed_model() -> None:
    config = load_command_settings("robustness-report")
    assert isinstance(config, RobustnessReportSettings)
    assert config.input.symbol == "BTCUSDT"
    assert config.evaluation.split_filter == ["test"]
    assert len(config.families) == 3
    assert config.families[0].name == "rule_based"
    assert config.feature_ablation.allow_degenerate_fallback is True
    assert len(config.feature_ablation.groups) >= 3


def test_evaluate_baseline_metadata_points_to_expected_default_file() -> None:
    settings = get_command_settings("evaluate-baseline")
    assert settings.default_config_path.name == "baseline.yaml"
    assert settings.config_model is BaselineSettings


def test_sync_vault_default_config_loads_typed_model() -> None:
    config = load_command_settings("sync-vault")
    assert isinstance(config, IntegrationSettings)
    assert config.input.signals_path.endswith("signals.parquet")
    assert config.input.leaderboard_path.endswith("leaderboard.parquet")
    assert config.contract.broadcast is False
    assert config.contract.update_nav is True


def test_run_demo_default_config_loads_typed_model() -> None:
    config = load_command_settings("run-demo")
    assert isinstance(config, DemoWorkflowSettings)
    assert config.commands.fetch_data_config_path.endswith("configs/data/default.yaml")
    assert config.commands.deep_learning_comparison_config_path.endswith(
        "configs/experiments/dl/regression_all.yaml"
    )
    assert config.stages.train_deep_learning.optional is True
    assert config.stages.compare_deep_learning.optional is True
    assert config.stages.generate_deep_learning_signals.optional is True
    assert config.output.run_name == "full_demo_default"


def test_compare_dl_default_config_loads_typed_model() -> None:
    config = load_command_settings("compare-dl")
    assert isinstance(config, DeepLearningComparisonSettings)
    assert config.experiment_name == "sequence_regression_all"
    assert len(config.runs) == 4
    assert config.runs[0].config_path.endswith("configs/models/lstm.yaml")


def test_exploratory_showcase_configs_load_typed_models() -> None:
    dataset = load_settings(
        Path("configs/models/exploratory_dl/dataset.yaml"),
        ExploratoryDLDatasetSettings,
    )
    gross = load_settings(
        Path("configs/experiments/dl/exploratory_gross_regression.yaml"),
        DeepLearningComparisonSettings,
    )
    direction = load_settings(
        Path("configs/experiments/dl/exploratory_direction_classification.yaml"),
        DeepLearningComparisonSettings,
    )
    signals = load_settings(
        Path("configs/signals/exploratory_dl/default.yaml"),
        ExploratoryDLSignalSettings,
    )
    report = load_settings(
        Path("configs/reports/exploratory_dl/showcase.yaml"),
        ExploratoryDLReportSettings,
    )
    workflow = load_command_settings("run-exploratory-dl-demo")

    assert dataset.output.output_dir == "data/processed/exploratory_dl"
    assert gross.experiment_name == "exploratory_gross_regression"
    assert len(gross.runs) == 4
    assert direction.experiment_name == "exploratory_direction_classification"
    assert direction.ranking.test_metric == "f1"
    assert len(signals.input.runs) == 8
    assert report.output.output_dir == "reports/exploratory_dl"
    assert isinstance(workflow, DemoWorkflowSettings)
    assert workflow.commands.exploratory_direction_comparison_config_path.endswith(
        "configs/experiments/dl/exploratory_direction_classification.yaml"
    )


def test_exploratory_outputs_are_separate_from_strict_defaults() -> None:
    strict_backtest = load_command_settings("backtest")
    exploratory_backtest = load_settings(
        Path("configs/backtests/exploratory_dl/default.yaml"),
        BacktestSettings,
    )
    strict_signals = load_command_settings("generate-signals")
    exploratory_signals = load_settings(
        Path("configs/signals/exploratory_dl/default.yaml"),
        ExploratoryDLSignalSettings,
    )

    assert exploratory_backtest.reporting.output_dir != strict_backtest.reporting.output_dir
    assert exploratory_backtest.input.signal_path != strict_backtest.input.signal_path
    assert exploratory_signals.output.output_dir != strict_signals.output.output_dir
