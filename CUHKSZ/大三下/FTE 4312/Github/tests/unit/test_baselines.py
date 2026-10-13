from __future__ import annotations

import pandas as pd
import pytest

from funding_arb.config.models import BaselineSettings, RuleBaselineSpec
from funding_arb.models.baselines import (
    _fit_classifier_with_calibration,
    _permutation_importance_frame,
    _rule_prediction_frame,
    _select_classifier_threshold,
    evaluate_prediction_table,
    make_time_series_folds,
    select_feature_columns,
)
from funding_arb.utils.degeneracy import DegenerateExperimentError


def _baseline_settings() -> BaselineSettings:
    return BaselineSettings.model_validate(
        {
            "input": {
                "dataset_path": "data/processed/supervised/binance/btcusdt/1h/btcusdt_supervised_dataset.parquet",
                "provider": "binance",
                "symbol": "BTCUSDT",
                "venue": "binance",
                "frequency": "1h",
            },
            "target": {
                "timestamp_column": "timestamp",
                "split_column": "split",
                "ready_column": "supervised_ready",
                "classification_column": "target_is_profitable_24h",
                "regression_column": "target_future_net_return_bps_24h",
            },
            "feature_selection": {
                "exclude_columns": [
                    "timestamp",
                    "split",
                    "feature_ready",
                    "supervised_ready",
                ],
                "exclude_prefixes": ["target_"],
                "max_missing_fraction": 0.4,
                "drop_constant_features": True,
            },
            "rules": [],
            "predictive": {},
            "output": {
                "model_dir": "data/artifacts/models/baselines",
                "run_name": "unit_test",
                "write_csv": True,
                "write_markdown_report": False,
            },
        }
    )


def test_select_feature_columns_excludes_targets_constants_and_sparse_columns() -> None:
    settings = _baseline_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "split": ["train", "train", "validation", "test"],
            "supervised_ready": [1, 1, 1, 1],
            "feature_ready": [1, 1, 1, 1],
            "funding_rate_bps": [0.1, 0.2, 0.3, 0.4],
            "spread_zscore_72h": [1.0, 0.5, 0.0, -0.5],
            "always_one": [1, 1, 1, 1],
            "mostly_missing": [1.0, None, None, None],
            "target_is_profitable_24h": [0, 1, 0, 0],
            "target_future_net_return_bps_24h": [1.0, 2.0, 3.0, 4.0],
        }
    )

    columns = select_feature_columns(frame, settings)

    assert columns == ["funding_rate_bps", "spread_zscore_72h"]


def test_rule_prediction_frame_applies_combined_threshold_and_regime_filter() -> None:
    settings = _baseline_settings()
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "split": ["train", "validation", "test"],
            "funding_rate_bps": [1.2, 0.8, 1.4],
            "spread_zscore_72h": [1.6, 1.8, 1.4],
            "positive_funding_regime": [1, 1, 1],
            "target_is_profitable_24h": [1, 0, 1],
            "target_future_net_return_bps_24h": [12.0, -4.0, 8.0],
        }
    )
    rule = RuleBaselineSpec(
        name="combined",
        kind="combined_threshold",
        funding_column="funding_rate_bps",
        funding_threshold_bps=1.0,
        spread_column="spread_zscore_72h",
        spread_threshold=1.5,
        regime_column="positive_funding_regime",
        regime_value=1,
    )

    predictions = _rule_prediction_frame(frame, settings, rule)

    assert predictions["signal"].tolist() == [1, 0, 0]
    assert predictions["predicted_label"].tolist() == [1, 0, 0]
    assert predictions["model_family"].unique().tolist() == ["rule_based"]


def test_evaluate_prediction_table_returns_split_metrics_for_classification_and_regression() -> None:
    predictions = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
            "split": ["train", "train", "validation", "validation", "test", "test"],
            "model_name": ["logit", "logit", "logit", "logit", "ridge", "ridge"],
            "model_family": ["linear", "linear", "linear", "linear", "linear", "linear"],
            "task": ["classification", "classification", "classification", "classification", "regression", "regression"],
            "signal_direction": ["short_perp_long_spot"] * 6,
            "signal": [1, 0, 1, 0, 1, 0],
            "decision_score": [0.8, 0.2, 0.6, 0.4, 10.0, -5.0],
            "signal_threshold": [0.5, 0.5, 0.5, 0.5, 0.0, 0.0],
            "signal_strength": [0.3, -0.3, 0.1, -0.1, 10.0, -5.0],
            "predicted_probability": [0.8, 0.2, 0.6, 0.4, None, None],
            "predicted_return_bps": [None, None, None, None, 10.0, -5.0],
            "predicted_label": [1, 0, 1, 0, 1, 0],
            "actual_label": [1, 0, 0, 0, 1, 0],
            "actual_return_bps": [5.0, -2.0, -1.0, -3.0, 8.0, -6.0],
        }
    )

    metrics = evaluate_prediction_table(predictions)

    assert set(metrics["task"].tolist()) == {"classification", "regression"}
    classification_row = metrics[(metrics["model_name"] == "logit") & (metrics["split"] == "train")].iloc[0]
    regression_row = metrics[(metrics["model_name"] == "ridge") & (metrics["split"] == "test")].iloc[0]
    assert classification_row["signal_count"] == 1
    assert regression_row["signal_count"] == 1
    assert regression_row["mae"] == 1.5


def test_make_time_series_folds_respects_gap_and_rolling_window() -> None:
    folds = make_time_series_folds(
        30,
        n_splits=3,
        gap=2,
        mode="rolling",
        min_train_size=10,
        rolling_window_size=8,
    )

    assert len(folds) == 3
    for train_index, validation_index in folds:
        assert len(train_index) <= 8
        assert train_index.max() < validation_index.min()
        assert validation_index.min() - train_index.max() >= 3


def test_select_classifier_threshold_prefers_higher_validation_signal_quality() -> None:
    settings = _baseline_settings()
    settings.threshold_search.objective = "avg_signal_return_bps"
    score_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "split": ["validation"] * 4,
            "model_name": ["logit"] * 4,
            "model_family": ["linear"] * 4,
            "task": ["classification"] * 4,
            "signal_direction": ["short_perp_long_spot"] * 4,
            "decision_score": [0.90, 0.70, 0.60, 0.40],
            "predicted_probability": [0.90, 0.70, 0.60, 0.40],
            "predicted_return_bps": [None, None, None, None],
            "actual_label": [1, 1, 0, 0],
            "actual_return_bps": [10.0, 1.0, -5.0, -1.0],
            "selected_hyperparameters_json": ["{}"] * 4,
            "selected_threshold_objective": [None] * 4,
            "calibration_method": ["none"] * 4,
            "feature_importance_method": ["permutation_validation"] * 4,
            "prediction_mode": ["static"] * 4,
        }
    )

    result = _select_classifier_threshold(
        score_frame,
        settings,
        model_name="logit",
        source="baseline",
        label_diagnostics_by_split={
            "validation": {
                "supports_threshold_selection": True,
                "tradeable_rate": 0.5,
                "profitable_rate": 0.5,
                "future_net_return_bps": {"min": -5.0, "max": 10.0, "mean": 1.25},
            }
        },
        default_threshold=0.5,
        threshold_grid=[0.5, 0.75],
    )

    assert result.selected_threshold == 0.75
    assert result.objective_value == 10.0
    assert result.search_frame.loc[result.search_frame["selected"], "threshold"].iloc[0] == 0.75


def test_select_classifier_threshold_raises_on_no_valid_candidates() -> None:
    settings = _baseline_settings()
    settings.threshold_search.objective = "avg_signal_return_bps"
    score_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "split": ["validation"] * 4,
            "model_name": ["logit"] * 4,
            "model_family": ["linear"] * 4,
            "task": ["classification"] * 4,
            "signal_direction": ["short_perp_long_spot"] * 4,
            "decision_score": [0.1, 0.2, 0.1, 0.2],
            "predicted_probability": [0.1, 0.2, 0.1, 0.2],
            "predicted_return_bps": [None, None, None, None],
            "actual_label": [0, 0, 0, 0],
            "actual_return_bps": [-2.0, -1.0, -3.0, -0.5],
            "selected_hyperparameters_json": ["{}"] * 4,
            "selected_threshold_objective": [None] * 4,
            "calibration_method": ["none"] * 4,
            "feature_importance_method": ["permutation_validation"] * 4,
            "prediction_mode": ["static"] * 4,
        }
    )

    with pytest.raises(DegenerateExperimentError, match="threshold_search"):
        _select_classifier_threshold(
            score_frame,
            settings,
            model_name="logit",
            source="baseline",
            label_diagnostics_by_split={
                "validation": {
                    "supports_threshold_selection": True,
                    "tradeable_rate": 0.25,
                    "profitable_rate": 0.25,
                    "future_net_return_bps": {"min": -3.0, "max": 1.0, "mean": -1.375},
                }
            },
            default_threshold=0.5,
            threshold_grid=[0.5, 0.75],
        )


def test_fit_classifier_with_calibration_returns_sigmoid_model_when_enabled() -> None:
    settings = _baseline_settings()
    settings.tuning.n_splits = 2
    settings.tuning.min_train_size = 6
    settings.tuning.gap = 0
    settings.predictive.classification.calibration_method = "sigmoid"
    settings.predictive.classification.calibration_cv_splits = 2

    train_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC"),
            "split": ["train"] * 12,
            "feature_1": [0.0, 0.1, 0.2, 0.0, 0.2, 0.3, 1.0, 1.1, 1.2, 0.9, 1.3, 1.4],
            "target_is_profitable_24h": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        }
    )

    model, diagnostic_model, calibration_used = _fit_classifier_with_calibration(
        settings,
        train_frame,
        ["feature_1"],
        "target_is_profitable_24h",
        settings.predictive.classification,
        {},
    )

    assert calibration_used == "sigmoid"
    assert hasattr(model, "predict_proba")
    assert hasattr(diagnostic_model, "named_steps")


def test_permutation_importance_frame_returns_ranked_features() -> None:
    settings = _baseline_settings()
    evaluation_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC"),
            "split": ["validation"] * 12,
            "feature_1": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "feature_2": [1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
            "target_future_net_return_bps_24h": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0],
            "target_is_profitable_24h": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        }
    )
    train_frame = evaluation_frame.copy()
    model = settings.predictive.regression
    from funding_arb.models.baselines import _fit_regressor

    regressor = _fit_regressor(
        settings,
        train_frame,
        ["feature_1", "feature_2"],
        "target_future_net_return_bps_24h",
        model,
        {},
    )

    importance = _permutation_importance_frame(
        regressor,
        evaluation_frame,
        ["feature_1", "feature_2"],
        settings,
        task="regression",
    )

    assert importance is not None
    assert set(importance["feature"].tolist()) == {"feature_1", "feature_2"}
