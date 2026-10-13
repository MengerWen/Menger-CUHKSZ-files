from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from funding_arb.config.models import DeepLearningSettings
from funding_arb.models.deep_learning import (
    GRUSequenceModel,
    LSTMSequenceModel,
    SequenceDataset,
    TCNSequenceModel,
    TransformerEncoderSequenceModel,
    _apply_threshold,
    _loss_function,
    _selection_resolution,
    _select_threshold,
    build_sequence_model,
    build_sequence_indices,
)
from funding_arb.utils.degeneracy import DegenerateExperimentError



def _settings(allow_cross_split_context: bool = True) -> DeepLearningSettings:
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
            "sequence": {
                "lookback_steps": 3,
                "allow_cross_split_context": allow_cross_split_context,
            },
            "model": {
                "name": "lstm",
                "hidden_size": 8,
                "num_layers": 1,
                "dropout": 0.0,
                "bidirectional": False,
            },
            "training": {
                "batch_size": 2,
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
                "model_dir": "data/artifacts/models/dl",
                "run_name": "unit_test",
                "write_csv": True,
                "write_markdown_report": False,
            },
        }
    )



def test_build_sequence_indices_respects_lookback_and_cross_split_context() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC"),
            "split": ["train", "train", "train", "validation", "validation", "test"],
            "supervised_ready": [1, 1, 1, 1, 1, 1],
            "target_future_net_return_bps_24h": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target_is_profitable_24h": [0, 0, 1, 1, 1, 1],
        }
    )

    with_cross = build_sequence_indices(frame, _settings(True))
    without_cross = build_sequence_indices(frame, _settings(False))

    assert with_cross == {"train": [2], "validation": [3, 4], "test": [5]}
    assert without_cross == {"train": [2], "validation": [], "test": []}



def test_sequence_dataset_returns_expected_historical_window() -> None:
    features = np.arange(24, dtype=np.float32).reshape(6, 4)
    targets = np.array([0, 1, 2, 3, 4, 5], dtype=np.float32)
    dataset = SequenceDataset(features=features, targets=targets, sample_indices=[2, 4], lookback_steps=3)

    sequence, target, row_index = dataset[1]

    assert sequence.shape == (3, 4)
    assert torch.equal(sequence, torch.tensor(features[2:5]))
    assert float(target.item()) == 4.0
    assert int(row_index.item()) == 4



def test_lstm_sequence_model_forward_shape_is_batch_scalar() -> None:
    model = LSTMSequenceModel(
        input_size=5,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
    )
    batch = torch.randn(4, 12, 5)

    output = model(batch)

    assert output.shape == (4,)


def test_gru_sequence_model_forward_shape_is_batch_scalar() -> None:
    model = GRUSequenceModel(
        input_size=5,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
        bidirectional=False,
    )
    batch = torch.randn(4, 12, 5)

    output = model(batch)

    assert output.shape == (4,)


def test_tcn_sequence_model_forward_shape_is_batch_scalar() -> None:
    model = TCNSequenceModel(
        input_size=5,
        hidden_channels=8,
        num_blocks=3,
        kernel_size=3,
        dilation_base=2,
        dropout=0.0,
        use_residual=True,
    )
    batch = torch.randn(4, 12, 5)

    output = model(batch)

    assert output.shape == (4,)


def test_transformer_encoder_sequence_model_forward_shape_is_batch_scalar() -> None:
    model = TransformerEncoderSequenceModel(
        input_size=5,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        pooling="last",
    )
    batch = torch.randn(4, 12, 5)

    output = model(batch)

    assert output.shape == (4,)


def test_build_sequence_model_dispatches_supported_architectures() -> None:
    settings = _settings()
    settings.model.name = "lstm"
    assert isinstance(build_sequence_model(5, settings), LSTMSequenceModel)

    settings.model.name = "gru"
    assert isinstance(build_sequence_model(5, settings), GRUSequenceModel)

    settings.model.name = "tcn"
    assert isinstance(build_sequence_model(5, settings), TCNSequenceModel)

    settings.model.name = "transformer_encoder"
    settings.model.transformer_d_model = 8
    settings.model.transformer_nhead = 2
    settings.model.transformer_dim_feedforward = 16
    assert isinstance(build_sequence_model(5, settings), TransformerEncoderSequenceModel)


def test_loss_function_supports_huber_regression() -> None:
    settings = _settings()
    settings.training.regression_loss = "huber"
    loss_fn, metadata = _loss_function(settings, torch.device("cpu"), np.array([1.0, -2.0], dtype=np.float32))

    assert loss_fn.__class__.__name__ == "HuberLoss"
    assert metadata["loss_name"] == "huber"


def test_threshold_selection_uses_validation_objective() -> None:
    settings = _settings()
    settings.threshold_search.enabled = True
    settings.threshold_search.objective = "avg_signal_return_bps"
    validation_scores = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "split": ["validation"] * 4,
            "model_name": ["lstm"] * 4,
            "model_family": ["deep_learning"] * 4,
            "task": ["regression"] * 4,
            "signal_direction": ["short_perp_long_spot"] * 4,
            "decision_score": [6.0, 3.0, -1.0, 0.5],
            "predicted_probability": [np.nan] * 4,
            "predicted_return_bps": [6.0, 3.0, -1.0, 0.5],
            "actual_label": [1, 1, 0, 0],
            "actual_return_bps": [4.0, 1.0, -2.0, -0.5],
            "selected_hyperparameters_json": ["{}"] * 4,
            "selected_threshold_objective": ["avg_signal_return_bps"] * 4,
            "calibration_method": ["none"] * 4,
            "feature_importance_method": ["ablation_validation"] * 4,
            "prediction_mode": ["static"] * 4,
        }
    )

    result = _select_threshold(
        validation_scores,
        settings,
        label_diagnostics_by_split={
            "validation": {
                "supports_threshold_selection": True,
                "tradeable_rate": 0.5,
                "profitable_rate": 0.5,
                "future_net_return_bps": {"min": -2.0, "max": 4.0, "mean": 0.875},
            }
        },
    )
    predictions = _apply_threshold(validation_scores, settings, result.selected_threshold)

    assert float(result.selected_threshold) in {-5.0, 0.0, 2.5, 5.0, 7.5, 10.0}
    assert result.objective_value is not None
    assert result.search_frame["selected"].sum() == 1
    assert "signal" in predictions.columns


def test_threshold_selection_raises_on_no_valid_candidates() -> None:
    settings = _settings()
    settings.threshold_search.enabled = True
    validation_scores = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "split": ["validation"] * 4,
            "model_name": ["lstm"] * 4,
            "model_family": ["deep_learning"] * 4,
            "task": ["regression"] * 4,
            "signal_direction": ["short_perp_long_spot"] * 4,
            "decision_score": [-6.0, -7.0, -8.0, -9.0],
            "predicted_probability": [np.nan] * 4,
            "predicted_return_bps": [-6.0, -7.0, -8.0, -9.0],
            "actual_label": [0, 0, 0, 0],
            "actual_return_bps": [-4.0, -1.0, -2.0, -0.5],
            "selected_hyperparameters_json": ["{}"] * 4,
            "selected_threshold_objective": ["avg_signal_return_bps"] * 4,
            "calibration_method": ["none"] * 4,
            "feature_importance_method": ["ablation_validation"] * 4,
            "prediction_mode": ["static"] * 4,
        }
    )

    with pytest.raises(DegenerateExperimentError, match="threshold_search"):
        _select_threshold(
            validation_scores,
            settings,
            label_diagnostics_by_split={
                "validation": {
                    "supports_threshold_selection": True,
                    "tradeable_rate": 0.25,
                    "profitable_rate": 0.25,
                    "future_net_return_bps": {"min": -4.0, "max": 1.0, "mean": -1.875},
                }
            },
        )


def test_selection_metric_falls_back_to_validation_loss_when_signal_metric_is_missing() -> None:
    selection = _selection_resolution(
        "validation_avg_signal_return_bps",
        validation_loss=1.25,
        validation_metrics={"avg_signal_return_bps": np.nan},
    )

    assert selection["configured_metric"] == "validation_avg_signal_return_bps"
    assert selection["configured_value"] is None
    assert selection["effective_metric"] == "validation_loss"
    assert selection["effective_value"] == 1.25
    assert selection["fallback_used"] is True


def test_tcn_encode_sequence_is_causal() -> None:
    model = TCNSequenceModel(
        input_size=3,
        hidden_channels=8,
        num_blocks=3,
        kernel_size=3,
        dilation_base=2,
        dropout=0.0,
        use_residual=True,
    )
    model.eval()
    batch_a = torch.randn(1, 8, 3)
    batch_b = batch_a.clone()
    batch_b[:, 5:, :] = batch_b[:, 5:, :] + 10.0

    encoded_a = model.encode_sequence(batch_a)
    encoded_b = model.encode_sequence(batch_b)

    assert torch.allclose(encoded_a[:, :5, :], encoded_b[:, :5, :], atol=1e-6)


def test_transformer_encoder_is_causal_over_sequence_positions() -> None:
    model = TransformerEncoderSequenceModel(
        input_size=3,
        d_model=8,
        nhead=2,
        num_layers=2,
        dim_feedforward=16,
        dropout=0.0,
        pooling="last",
    )
    model.eval()
    batch_a = torch.randn(1, 8, 3)
    batch_b = batch_a.clone()
    batch_b[:, 5:, :] = batch_b[:, 5:, :] + 10.0

    encoded_a = model.encode_sequence(batch_a)
    encoded_b = model.encode_sequence(batch_b)

    assert torch.allclose(encoded_a[:, :5, :], encoded_b[:, :5, :], atol=1e-6)
