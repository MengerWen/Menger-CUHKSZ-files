"""Sequence-model deep-learning pipeline for the funding-rate arbitrage project."""

from __future__ import annotations

import copy
import itertools
import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from funding_arb.config.models import DeepLearningSettings
from funding_arb.models.baselines import evaluate_prediction_table
from funding_arb.utils.degeneracy import (
    DegenerateExperimentDiagnostics,
    DegenerateExperimentError,
    infer_horizon_label,
    infer_profitable_column,
    infer_tradeable_column,
    label_split_diagnostics,
    signal_split_diagnostics,
    summarize_threshold_search,
    warn_on_degenerate_experiment,
)
from funding_arb.utils.paths import ensure_directory, repo_path

DEFAULT_PROBABILITY_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
DEFAULT_REGRESSION_THRESHOLD_GRID = [-5.0, 0.0, 2.5, 5.0, 7.5, 10.0]
SELECTION_METRIC_DIRECTIONS = {
    "validation_loss": "min",
    "validation_f1": "max",
    "validation_roc_auc": "max",
    "validation_pearson_corr": "max",
    "validation_avg_signal_return_bps": "max",
    "validation_signal_hit_rate": "max",
    "validation_cumulative_signal_return_bps": "max",
}


@dataclass(frozen=True)
class NormalizationStats:
    """Feature preprocessing parameters fitted on historical training data only."""

    medians: dict[str, float]
    centers: dict[str, float]
    scales: dict[str, float]
    lower_bounds: dict[str, float] | None
    upper_bounds: dict[str, float] | None
    scaler: str
    winsorize_lower_quantile: float | None
    winsorize_upper_quantile: float | None


@dataclass(frozen=True)
class DeepLearningArtifacts:
    """Paths produced by the deep-learning training pipeline."""

    output_dir: str
    checkpoint_path: str
    history_path: str
    predictions_path: str
    predictions_csv_path: str | None
    metrics_path: str
    metrics_csv_path: str | None
    leaderboard_path: str
    leaderboard_csv_path: str | None
    report_path: str | None
    manifest_path: str
    feature_columns_path: str
    normalization_path: str
    diagnostic_paths: dict[str, str]


@dataclass(frozen=True)
class ThresholdSelectionResult:
    """Outcome of selecting a decision threshold on validation data."""

    selected_threshold: float
    objective_value: float | None
    search_frame: pd.DataFrame
    fallback_used: bool = False
    fallback_reason: str | None = None


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazy sliding-window dataset that slices only historical context for each target row."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sample_indices: list[int],
        lookback_steps: int,
    ) -> None:
        self.features = features
        self.targets = targets
        self.sample_indices = sample_indices
        self.lookback_steps = lookback_steps

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_index = self.sample_indices[item]
        start_index = target_index - self.lookback_steps + 1
        sequence = torch.from_numpy(self.features[start_index : target_index + 1]).float()
        target = torch.tensor(self.targets[target_index], dtype=torch.float32)
        row_index = torch.tensor(target_index, dtype=torch.long)
        return sequence, target, row_index


class LSTMSequenceModel(nn.Module):
    """Simple LSTM encoder with a single scalar prediction head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(output_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, (hidden_state, _) = self.encoder(inputs)
        if self.bidirectional:
            encoded = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
        else:
            encoded = hidden_state[-1]
        return self.head(self.dropout(encoded)).squeeze(-1)


class GRUSequenceModel(nn.Module):
    """Simple GRU encoder with the same scalar prediction head shape as the LSTM."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        output_size = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(output_size, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden_state = self.encoder(inputs)
        if self.bidirectional:
            encoded = torch.cat([hidden_state[-2], hidden_state[-1]], dim=1)
        else:
            encoded = hidden_state[-1]
        return self.head(self.dropout(encoded)).squeeze(-1)


class CausalConv1d(nn.Module):
    """One-dimensional convolution with left padding only so time remains causal."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.left_padding, 0))
        return self.conv(padded)


class TCNResidualBlock(nn.Module):
    """Compact residual TCN block with causal dilated convolutions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.conv2 = CausalConv1d(
            out_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if use_residual and in_channels != out_channels
            else nn.Identity()
        )
        self.use_residual = use_residual

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.residual(inputs)
        outputs = self.conv1(inputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        outputs = self.conv2(outputs)
        outputs = self.activation(outputs)
        outputs = self.dropout(outputs)
        if self.use_residual:
            outputs = outputs + residual
        return outputs


class TCNSequenceModel(nn.Module):
    """Lightweight causal temporal convolution network with a scalar output head."""

    def __init__(
        self,
        input_size: int,
        hidden_channels: int,
        num_blocks: int,
        kernel_size: int,
        dilation_base: int,
        dropout: float,
        use_residual: bool,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = input_size
        for block_index in range(num_blocks):
            dilation = int(dilation_base ** block_index)
            blocks.append(
                TCNResidualBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    use_residual=use_residual,
                )
            )
            in_channels = hidden_channels
        self.network = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_channels, 1)

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs.transpose(1, 2)
        for block in self.network:
            outputs = block(outputs)
        return outputs.transpose(1, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_sequence = self.encode_sequence(inputs)
        encoded = encoded_sequence[:, -1, :]
        return self.head(self.dropout(encoded)).squeeze(-1)


def _sinusoidal_positional_encoding(
    sequence_length: int,
    d_model: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(sequence_length, device=device, dtype=dtype).unsqueeze(1)
    div_terms = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype)
        * (-math.log(10_000.0) / max(d_model, 1))
    )
    encoding = torch.zeros(sequence_length, d_model, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(positions * div_terms)
    encoding[:, 1::2] = torch.cos(positions * div_terms[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class TransformerEncoderSequenceModel(nn.Module):
    """Compact causal TransformerEncoder with final-token or mean pooling."""

    def __init__(
        self,
        input_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.input_projection = nn.Linear(input_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, 1)

    @staticmethod
    def causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(sequence_length, sequence_length, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def encode_sequence(self, inputs: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(inputs)
        projected = projected + _sinusoidal_positional_encoding(
            projected.shape[1],
            projected.shape[2],
            device=projected.device,
            dtype=projected.dtype,
        )
        mask = self.causal_mask(projected.shape[1], projected.device)
        return self.encoder(projected, mask=mask)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_sequence = self.encode_sequence(inputs)
        if self.pooling == "mean":
            encoded = encoded_sequence.mean(dim=1)
        else:
            encoded = encoded_sequence[:, -1, :]
        return self.head(self.dropout(encoded)).squeeze(-1)


def describe_deep_learning_job(config: DeepLearningSettings | dict[str, Any]) -> str:
    """Return a human-readable summary of the deep-learning training job."""
    settings = (
        config if isinstance(config, DeepLearningSettings) else DeepLearningSettings.model_validate(config)
    )
    return (
        f"Deep-learning training ready for {settings.input.symbol} on {settings.input.provider} at "
        f"{settings.input.frequency}, task={settings.target.task}, target={settings.target.column}, "
        f"lookback={settings.sequence.lookback_steps}, model={settings.model.name}, "
        f"selection_metric={settings.training.selection_metric}, prediction_mode={settings.prediction.mode}. "
        f"Artifacts will be written under {settings.output.model_dir}/{settings.input.provider}/"
        f"{settings.input.symbol.lower()}/{settings.input.frequency}/{settings.output.run_name}."
    )


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return repo_path(*path.parts)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(converted) or math.isinf(converted):
        return None
    return converted


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    text = str(value).strip()
    return text or None


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _table_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no rows)"
    try:
        return frame.to_markdown(index=False)
    except Exception:
        return frame.to_string(index=False)


def _load_supervised_dataset(settings: DeepLearningSettings) -> pd.DataFrame:
    dataset_path = _resolve_path(settings.input.dataset_path)
    suffix = dataset_path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(dataset_path)
    elif suffix == ".csv":
        frame = pd.read_csv(dataset_path)
    else:
        raise ValueError(f"Unsupported supervised dataset format: {dataset_path.suffix}")
    timestamp_column = settings.target.timestamp_column
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], utc=True)
    return frame.sort_values(timestamp_column).reset_index(drop=True)


def select_feature_columns(frame: pd.DataFrame, settings: DeepLearningSettings) -> list[str]:
    """Select leakage-safe numeric features for sequence modeling."""
    feature_settings = settings.feature_selection
    if feature_settings.include_columns:
        columns = [column for column in feature_settings.include_columns if column in frame.columns]
    else:
        columns = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    excluded_columns = set(feature_settings.exclude_columns)
    columns = [
        column
        for column in columns
        if column not in excluded_columns
        and not any(column.startswith(prefix) for prefix in feature_settings.exclude_prefixes)
    ]
    if not columns:
        raise ValueError("Feature selection removed every candidate feature column for deep learning.")

    split_column = settings.target.split_column
    train_frame = frame[frame[split_column] == "train"].copy()
    candidate_frame = train_frame[columns].replace([np.inf, -np.inf], np.nan)
    missing_fraction = candidate_frame.isna().mean()
    columns = [
        column
        for column in columns
        if float(missing_fraction[column]) <= float(feature_settings.max_missing_fraction)
    ]
    if feature_settings.drop_constant_features:
        filtered_frame = candidate_frame[columns]
        columns = [column for column in columns if int(filtered_frame[column].nunique(dropna=True)) > 1]
    if not columns:
        raise ValueError("No usable feature columns remain for deep learning after filtering.")
    return sorted(columns)


def fit_normalization_stats(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_column: str,
    settings: DeepLearningSettings | None = None,
    fit_row_indices: list[int] | None = None,
) -> NormalizationStats:
    """Fit preprocessing statistics on historical rows only."""
    if fit_row_indices is None:
        fit_frame = frame.loc[frame[split_column] == "train", feature_columns].copy()
    else:
        fit_frame = frame.iloc[fit_row_indices][feature_columns].copy()
    fit_frame = fit_frame.replace([np.inf, -np.inf], np.nan).astype(float)
    medians = fit_frame.median(skipna=True).fillna(0.0)
    filled = fit_frame.fillna(medians)

    preprocessing = settings.preprocessing if settings is not None else None
    lower_quantile = preprocessing.winsorize_lower_quantile if preprocessing is not None else None
    upper_quantile = preprocessing.winsorize_upper_quantile if preprocessing is not None else None
    lower_bounds: dict[str, float] | None = None
    upper_bounds: dict[str, float] | None = None
    clipped = filled
    if lower_quantile is not None and upper_quantile is not None:
        lower = filled.quantile(lower_quantile).fillna(medians)
        upper = filled.quantile(upper_quantile).fillna(medians)
        clipped = filled.clip(lower=lower, upper=upper, axis=1)
        lower_bounds = {column: float(value) for column, value in lower.items()}
        upper_bounds = {column: float(value) for column, value in upper.items()}

    scaler = preprocessing.scaler if preprocessing is not None else "standard"
    if scaler == "robust":
        centers = clipped.median().fillna(0.0)
        q75 = clipped.quantile(0.75)
        q25 = clipped.quantile(0.25)
        scales = (q75 - q25).replace(0.0, 1.0).fillna(1.0)
    else:
        centers = clipped.mean().fillna(0.0)
        scales = clipped.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

    return NormalizationStats(
        medians={column: float(value) for column, value in medians.items()},
        centers={column: float(value) for column, value in centers.items()},
        scales={column: float(value) for column, value in scales.items()},
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        scaler=scaler,
        winsorize_lower_quantile=lower_quantile,
        winsorize_upper_quantile=upper_quantile,
    )


def transform_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    stats: NormalizationStats,
) -> np.ndarray:
    """Impute, optionally winsorize, and scale features using training-only statistics."""
    feature_frame = frame[feature_columns].replace([np.inf, -np.inf], np.nan).astype(float).copy()
    medians = pd.Series(stats.medians)
    centers = pd.Series(stats.centers)
    scales = pd.Series(stats.scales)
    feature_frame = feature_frame.fillna(medians)
    if stats.lower_bounds is not None and stats.upper_bounds is not None:
        lower = pd.Series(stats.lower_bounds)
        upper = pd.Series(stats.upper_bounds)
        feature_frame = feature_frame.clip(lower=lower, upper=upper, axis=1)
    normalized = (feature_frame - centers) / scales
    return normalized.astype("float32").to_numpy()


def _filter_sequence_indices(
    frame: pd.DataFrame,
    candidate_indices: list[int],
    settings: DeepLearningSettings,
    *,
    require_same_split_context: bool,
) -> list[int]:
    lookback_steps = settings.sequence.lookback_steps
    split_column = settings.target.split_column
    ready_column = settings.target.ready_column
    target_column = settings.target.column

    target_values = pd.to_numeric(frame[target_column], errors="coerce")
    split_values = frame[split_column].astype(str)
    ready_values = frame[ready_column].fillna(0).astype(int)
    eligible: list[int] = []

    for row_index in candidate_indices:
        if row_index < lookback_steps - 1:
            continue
        if ready_values.iloc[row_index] != 1:
            continue
        if pd.isna(target_values.iloc[row_index]):
            continue
        if require_same_split_context and not settings.sequence.allow_cross_split_context:
            start_index = row_index - lookback_steps + 1
            context_splits = split_values.iloc[start_index : row_index + 1]
            if not context_splits.eq(split_values.iloc[row_index]).all():
                continue
        eligible.append(int(row_index))
    return eligible


def build_sequence_indices(frame: pd.DataFrame, settings: DeepLearningSettings) -> dict[str, list[int]]:
    """Build split-specific target indices using only historical lookback windows."""
    split_column = settings.target.split_column
    split_values = frame[split_column].astype(str)
    sample_indices: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for split_name in sample_indices:
        candidate_indices = split_values[split_values == split_name].index.tolist()
        sample_indices[split_name] = _filter_sequence_indices(
            frame,
            candidate_indices,
            settings,
            require_same_split_context=True,
        )
    return sample_indices


def _set_random_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def build_sequence_model(input_size: int, settings: DeepLearningSettings) -> nn.Module:
    """Dispatch to the configured sequence model implementation."""
    model_name = settings.model.name.lower()
    common_kwargs = {
        "input_size": input_size,
        "hidden_size": settings.model.hidden_size,
        "num_layers": settings.model.num_layers,
        "dropout": settings.model.dropout,
        "bidirectional": settings.model.bidirectional,
    }
    if model_name == "lstm":
        return LSTMSequenceModel(**common_kwargs)
    if model_name == "gru":
        return GRUSequenceModel(**common_kwargs)
    if model_name == "tcn":
        return TCNSequenceModel(
            input_size=input_size,
            hidden_channels=settings.model.tcn_hidden_channels,
            num_blocks=settings.model.tcn_num_blocks,
            kernel_size=settings.model.tcn_kernel_size,
            dilation_base=settings.model.tcn_dilation_base,
            dropout=settings.model.dropout,
            use_residual=settings.model.tcn_use_residual,
        )
    if model_name == "transformer_encoder":
        return TransformerEncoderSequenceModel(
            input_size=input_size,
            d_model=settings.model.transformer_d_model,
            nhead=settings.model.transformer_nhead,
            num_layers=settings.model.transformer_num_layers,
            dim_feedforward=settings.model.transformer_dim_feedforward,
            dropout=settings.model.dropout,
            pooling=settings.model.transformer_pooling,
        )
    raise ValueError(
        "Unsupported deep-learning model "
        f"'{settings.model.name}'. Supported models: 'lstm', 'gru', 'tcn', 'transformer_encoder'."
    )


def _loss_function(
    settings: DeepLearningSettings,
    device: torch.device,
    train_targets: np.ndarray,
) -> tuple[nn.Module, dict[str, Any]]:
    if settings.target.task == "classification":
        if settings.training.use_balanced_classification_loss:
            positives = float((train_targets > 0.5).sum())
            negatives = float((train_targets <= 0.5).sum())
            if positives > 0.0 and negatives > 0.0:
                pos_weight_value = negatives / positives
                pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
                return nn.BCEWithLogitsLoss(pos_weight=pos_weight), {
                    "loss_name": "bce_with_logits",
                    "pos_weight": float(pos_weight_value),
                    "use_balanced_classification_loss": True,
                }
        return nn.BCEWithLogitsLoss(), {
            "loss_name": "bce_with_logits",
            "pos_weight": None,
            "use_balanced_classification_loss": False,
        }
    if settings.target.task == "regression":
        if settings.training.regression_loss == "mse":
            return nn.MSELoss(), {"loss_name": "mse", "huber_delta": None}
        if settings.training.regression_loss == "huber":
            return nn.HuberLoss(delta=settings.training.huber_delta), {
                "loss_name": "huber",
                "huber_delta": float(settings.training.huber_delta),
            }
        if settings.training.regression_loss == "smooth_l1":
            return nn.SmoothL1Loss(beta=settings.training.huber_delta), {
                "loss_name": "smooth_l1",
                "huber_delta": float(settings.training.huber_delta),
            }
    raise ValueError(f"Unsupported deep-learning task '{settings.target.task}'.")


def _dataset_from_indices(
    features: np.ndarray,
    targets: np.ndarray,
    sample_indices: list[int],
    lookback_steps: int,
) -> SequenceDataset:
    return SequenceDataset(
        features=features,
        targets=targets,
        sample_indices=sample_indices,
        lookback_steps=lookback_steps,
    )


def _build_loader(
    dataset: SequenceDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def _run_loader(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: Adam | None,
    clip_grad_norm: float | None,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = 0.0
    sample_count = 0
    all_outputs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_row_indices: list[np.ndarray] = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for sequences, targets, row_indices in loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            outputs = model(sequences)
            loss = loss_fn(outputs, targets)

            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad_norm is not None and clip_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
                optimizer.step()

            batch_size = int(targets.shape[0])
            total_loss += float(loss.detach().item()) * batch_size
            sample_count += batch_size
            all_outputs.append(outputs.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())
            all_row_indices.append(row_indices.detach().cpu().numpy())

    if sample_count == 0:
        return {
            "loss": None,
            "outputs": np.array([], dtype=float),
            "targets": np.array([], dtype=float),
            "row_indices": np.array([], dtype=int),
        }

    return {
        "loss": total_loss / sample_count,
        "outputs": np.concatenate(all_outputs).astype(float),
        "targets": np.concatenate(all_targets).astype(float),
        "row_indices": np.concatenate(all_row_indices).astype(int),
    }


def _default_threshold(settings: DeepLearningSettings) -> float:
    if settings.target.task == "classification":
        return float(settings.target.probability_threshold)
    return float(settings.target.trade_threshold_bps)


def _default_threshold_grid(settings: DeepLearningSettings) -> list[float]:
    if settings.target.task == "classification":
        return settings.threshold_search.probability_grid or DEFAULT_PROBABILITY_GRID
    return settings.threshold_search.regression_threshold_grid_bps or DEFAULT_REGRESSION_THRESHOLD_GRID


def _split_metric_map(
    diagnostics_by_split: dict[str, dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    return {
        split_name: diagnostics.get(key)
        for split_name, diagnostics in diagnostics_by_split.items()
    }


def _return_summary_map(
    diagnostics_by_split: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    return {
        split_name: diagnostics.get("future_net_return_bps", {})
        for split_name, diagnostics in diagnostics_by_split.items()
    }


def _label_diagnostics_for_frame(
    frame: pd.DataFrame,
    settings: DeepLearningSettings,
) -> dict[str, dict[str, Any]]:
    tradeable_threshold_bps = 5.0
    profitable_threshold_bps = 0.0
    if settings.input.manifest_path:
        manifest_path = _resolve_path(settings.input.manifest_path)
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            target_config = manifest.get("target", {})
            tradeable_threshold_bps = float(
                target_config.get("min_expected_edge_bps", tradeable_threshold_bps)
            )
            profitable_threshold_bps = float(
                target_config.get("positive_return_threshold_bps", profitable_threshold_bps)
            )
    return label_split_diagnostics(
        frame,
        split_column=settings.target.split_column,
        net_return_column=settings.target.regression_column,
        tradeable_column=infer_tradeable_column(settings.target.regression_column),
        profitable_column=infer_profitable_column(settings.target.regression_column),
        tradeable_threshold_bps=tradeable_threshold_bps,
        profitable_threshold_bps=profitable_threshold_bps,
    )


def _finalize_threshold_search_frame(
    search_frame: pd.DataFrame,
    *,
    selected_threshold: float,
    status: str,
    reason: str | None,
    fallback_used: bool,
    fallback_reason: str | None,
) -> pd.DataFrame:
    frame = search_frame.copy()
    if frame.empty:
        frame = pd.DataFrame([{"threshold": float(selected_threshold), "objective_value": None}])
    if "valid_candidate" not in frame.columns:
        frame["valid_candidate"] = pd.to_numeric(frame["objective_value"], errors="coerce").notna()
    frame["selected"] = pd.to_numeric(frame["threshold"], errors="coerce").eq(float(selected_threshold))
    frame["status"] = status
    frame["reason"] = reason
    frame["fallback_used"] = bool(fallback_used)
    frame["fallback_reason"] = fallback_reason
    return frame


def _threshold_search_failure_result(
    *,
    settings: DeepLearningSettings,
    reason: str,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    signal_count_by_split: dict[str, int],
    default_threshold: float,
    search_frame: pd.DataFrame,
) -> ThresholdSelectionResult:
    summary = summarize_threshold_search(search_frame)
    diagnostics = DegenerateExperimentDiagnostics(
        stage="threshold_search",
        reason=reason,
        status="threshold_selection_failed",
        model_name=settings.model.name,
        source="deep_learning",
        target_horizon=infer_horizon_label(settings.target.regression_column),
        split="validation",
        selected_threshold=float(default_threshold),
        candidate_threshold_count=summary.get("candidate_count"),
        valid_candidate_count=summary.get("valid_candidate_count"),
        signal_count_by_split=signal_count_by_split,
        tradeable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "tradeable_rate"),
        profitable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "profitable_rate"),
        future_net_return_bps_by_split=_return_summary_map(label_diagnostics_by_split),
        fallback_used=bool(settings.threshold_search.allow_degenerate_fallback),
        fallback_reason=reason if settings.threshold_search.allow_degenerate_fallback else None,
        extra={"threshold_search_summary": summary},
    )
    if not settings.threshold_search.allow_degenerate_fallback:
        raise DegenerateExperimentError(diagnostics)
    warn_on_degenerate_experiment(diagnostics, stacklevel=3)
    finalized = _finalize_threshold_search_frame(
        search_frame,
        selected_threshold=float(default_threshold),
        status="degenerate_fallback",
        reason=reason,
        fallback_used=True,
        fallback_reason=reason,
    )
    return ThresholdSelectionResult(
        selected_threshold=float(default_threshold),
        objective_value=None,
        search_frame=finalized,
        fallback_used=True,
        fallback_reason=reason,
    )


def _threshold_search_success_result(
    selected_threshold: float,
    objective_value: float | None,
    search_frame: pd.DataFrame,
) -> ThresholdSelectionResult:
    return ThresholdSelectionResult(
        selected_threshold=float(selected_threshold),
        objective_value=_safe_float(objective_value),
        search_frame=_finalize_threshold_search_frame(
            search_frame,
            selected_threshold=float(selected_threshold),
            status="ok",
            reason=None,
            fallback_used=False,
            fallback_reason=None,
        ),
        fallback_used=False,
        fallback_reason=None,
    )


def _warn_on_zero_signal_splits(
    *,
    settings: DeepLearningSettings,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    predictions: pd.DataFrame,
) -> DegenerateExperimentDiagnostics | None:
    signal_diagnostics_by_split = signal_split_diagnostics(
        predictions,
        signal_column="signal",
        split_names=predictions["split"].dropna().astype(str).unique().tolist(),
    )
    flagged = [
        (split_name, diagnostics)
        for split_name, diagnostics in signal_diagnostics_by_split.items()
        if split_name in {"validation", "test"} and diagnostics.get("status") != "ok"
    ]
    if not flagged:
        return None
    reason = "; ".join(
        f"{split_name}: {diagnostics.get('reason') or diagnostics.get('status')}"
        for split_name, diagnostics in flagged
    )
    diagnostics = DegenerateExperimentDiagnostics(
        stage="signal_generation",
        reason=reason,
        status="no_tradable_signals",
        model_name=settings.model.name,
        source="deep_learning",
        target_horizon=infer_horizon_label(settings.target.regression_column),
        signal_count_by_split={
            split_name: int(split_diagnostics.get("signal_count", 0))
            for split_name, split_diagnostics in signal_diagnostics_by_split.items()
        },
        tradeable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "tradeable_rate"),
        profitable_rate_by_split=_split_metric_map(label_diagnostics_by_split, "profitable_rate"),
        future_net_return_bps_by_split=_return_summary_map(label_diagnostics_by_split),
    )
    warn_on_degenerate_experiment(diagnostics, stacklevel=3)
    return diagnostics


def _score_frame(
    frame: pd.DataFrame,
    row_indices: np.ndarray,
    outputs: np.ndarray,
    settings: DeepLearningSettings,
    *,
    prediction_mode: str,
    selected_hyperparameters_json: str,
    threshold_objective: str | None,
    feature_importance_method: str,
) -> pd.DataFrame:
    subset = frame.iloc[row_indices].copy().reset_index(drop=True)
    task = settings.target.task
    if task == "classification":
        predicted_probability = 1.0 / (1.0 + np.exp(-outputs))
        decision_score = predicted_probability.astype(float)
        predicted_return_bps = np.full(len(subset), np.nan)
    else:
        decision_score = outputs.astype(float)
        predicted_probability = np.full(len(subset), np.nan)
        predicted_return_bps = outputs.astype(float)

    return pd.DataFrame(
        {
            "timestamp": subset[settings.target.timestamp_column],
            "split": subset[settings.target.split_column],
            "model_name": settings.model.name,
            "model_family": "deep_learning",
            "task": task,
            "signal_direction": "short_perp_long_spot",
            "decision_score": decision_score,
            "predicted_probability": predicted_probability,
            "predicted_return_bps": predicted_return_bps,
            "actual_label": pd.to_numeric(subset[settings.target.classification_column], errors="coerce"),
            "actual_return_bps": pd.to_numeric(subset[settings.target.regression_column], errors="coerce"),
            "selected_hyperparameters_json": selected_hyperparameters_json,
            "selected_threshold_objective": threshold_objective,
            "calibration_method": "none",
            "feature_importance_method": feature_importance_method,
            "prediction_mode": prediction_mode,
        }
    )


def _apply_threshold(score_frame: pd.DataFrame, settings: DeepLearningSettings, threshold: float) -> pd.DataFrame:
    predictions = score_frame.copy()
    if settings.target.task == "classification":
        probabilities = pd.to_numeric(predictions["predicted_probability"], errors="coerce")
        predicted_label = (probabilities >= float(threshold)).astype(int)
        signal_strength = probabilities - float(threshold)
    else:
        predicted_return = pd.to_numeric(predictions["predicted_return_bps"], errors="coerce")
        predicted_label = (predicted_return >= float(threshold)).astype(int)
        signal_strength = predicted_return - float(threshold)
    predictions["signal"] = predicted_label.astype(int)
    predictions["predicted_label"] = predicted_label.astype(int)
    predictions["signal_threshold"] = float(threshold)
    predictions["signal_strength"] = signal_strength.astype(float)
    return predictions


def _metric_value_for_threshold(
    predictions: pd.DataFrame,
    *,
    objective: str,
    top_quantile: float,
) -> float:
    metrics = evaluate_prediction_table(predictions, top_quantile=top_quantile)
    if metrics.empty:
        return -np.inf
    value = metrics.iloc[0].get(objective)
    safe_value = _safe_float(value)
    return -np.inf if safe_value is None else float(safe_value)


def _select_threshold(
    validation_score_frame: pd.DataFrame,
    settings: DeepLearningSettings,
    *,
    label_diagnostics_by_split: dict[str, dict[str, Any]],
) -> ThresholdSelectionResult:
    default_threshold = _default_threshold(settings)
    candidates = sorted({float(value) for value in _default_threshold_grid(settings)} | {float(default_threshold)})
    if not settings.threshold_search.enabled:
        return _threshold_search_success_result(
            float(default_threshold),
            None,
            pd.DataFrame([{"threshold": float(default_threshold), "objective_value": None}]),
        )
    validation_support = label_diagnostics_by_split.get("validation", {})
    if validation_score_frame.empty:
        return _threshold_search_failure_result(
            settings=settings,
            reason="Validation split produced no score rows, so threshold selection cannot run.",
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={"validation": 0},
            default_threshold=float(default_threshold),
            search_frame=pd.DataFrame(),
        )
    if not bool(validation_support.get("supports_threshold_selection", True)):
        return _threshold_search_failure_result(
            settings=settings,
            reason=(
                "Validation split cannot support threshold selection because label diagnostics are "
                f"degenerate: {validation_support.get('reason') or validation_support.get('status')}."
            ),
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={"validation": 0},
            default_threshold=float(default_threshold),
            search_frame=pd.DataFrame(),
        )

    rows: list[dict[str, Any]] = []
    best_threshold = float(default_threshold)
    best_score = -np.inf
    for threshold in candidates:
        candidate_predictions = _apply_threshold(validation_score_frame, settings, threshold)
        objective_value = _metric_value_for_threshold(
            candidate_predictions,
            objective=settings.threshold_search.objective,
            top_quantile=settings.threshold_search.top_quantile,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "objective_value": _safe_float(objective_value),
                "valid_candidate": _safe_float(objective_value) is not None,
            }
        )
        if objective_value > best_score:
            best_score = objective_value
            best_threshold = float(threshold)
    search_frame = pd.DataFrame(rows)
    signal_diagnostics = signal_split_diagnostics(
        _apply_threshold(validation_score_frame, settings, best_threshold),
        signal_column="signal",
        split_names=["validation"],
    )
    if int(search_frame["valid_candidate"].sum()) == 0:
        return _threshold_search_failure_result(
            settings=settings,
            reason="Threshold search found no valid candidate on the validation split.",
            label_diagnostics_by_split=label_diagnostics_by_split,
            signal_count_by_split={
                split_name: int(diagnostics.get("signal_count", 0))
                for split_name, diagnostics in signal_diagnostics.items()
            },
            default_threshold=float(default_threshold),
            search_frame=search_frame,
        )
    return _threshold_search_success_result(best_threshold, best_score, search_frame)


def _metric_row(predictions: pd.DataFrame, settings: DeepLearningSettings) -> dict[str, Any]:
    metrics = evaluate_prediction_table(predictions, top_quantile=settings.threshold_search.top_quantile)
    if metrics.empty:
        return {}
    return metrics.iloc[0].to_dict()


def _selection_value(
    metric_name: str,
    validation_loss: float | None,
    validation_metrics: dict[str, Any],
) -> float | None:
    if metric_name == "validation_loss":
        return _safe_float(validation_loss)
    metric_key = metric_name.removeprefix("validation_")
    return _safe_float(validation_metrics.get(metric_key))


def _selection_resolution(
    metric_name: str,
    validation_loss: float | None,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Resolve checkpoint selection score, falling back to validation loss when needed.

    Strategy-oriented metrics can be undefined on a split with zero traded signals.
    In that case we fall back to validation loss so early stopping and best-checkpoint
    selection remain deterministic rather than drifting to the final epoch.
    """
    configured_value = _selection_value(metric_name, validation_loss, validation_metrics)
    if configured_value is not None:
        return {
            "configured_metric": metric_name,
            "configured_value": configured_value,
            "effective_metric": metric_name,
            "effective_value": configured_value,
            "fallback_used": False,
        }

    fallback_value = _safe_float(validation_loss)
    return {
        "configured_metric": metric_name,
        "configured_value": None,
        "effective_metric": "validation_loss",
        "effective_value": fallback_value,
        "fallback_used": fallback_value is not None and metric_name != "validation_loss",
    }


def _is_better_metric(candidate: float | None, best: float | None, metric_name: str) -> bool:
    if candidate is None:
        return False
    if best is None:
        return True
    direction = SELECTION_METRIC_DIRECTIONS.get(metric_name, "max")
    if direction == "min":
        return float(candidate) < float(best)
    return float(candidate) > float(best)


def _history_split_indices(
    history_indices: list[int],
    settings: DeepLearningSettings,
) -> tuple[list[int], list[int]]:
    if not history_indices:
        return [], []
    if len(history_indices) < max(settings.sequence.lookback_steps + 1, 8):
        return history_indices, []
    validation_size = max(1, int(len(history_indices) * settings.training.internal_validation_fraction))
    validation_size = min(validation_size, max(1, len(history_indices) // 3))
    train_indices = history_indices[:-validation_size]
    validation_indices = history_indices[-validation_size:]
    if len(train_indices) < settings.sequence.lookback_steps:
        return history_indices, []
    return train_indices, validation_indices


def _feature_group_catalog(
    settings: DeepLearningSettings,
    feature_columns: list[str],
) -> dict[str, list[str]]:
    manifest_path = settings.input.manifest_path
    if manifest_path is not None:
        manifest_file = _resolve_path(manifest_path)
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            feature_manifest_path = manifest.get("input", {}).get("feature_manifest_path")
            if feature_manifest_path:
                feature_manifest_file = _resolve_path(feature_manifest_path)
                if feature_manifest_file.exists():
                    feature_manifest = json.loads(feature_manifest_file.read_text(encoding="utf-8"))
                    groups = feature_manifest.get("feature_group_columns")
                    if isinstance(groups, dict):
                        filtered = {
                            str(group_name): sorted(
                                [column for column in columns if column in set(feature_columns)]
                            )
                            for group_name, columns in groups.items()
                        }
                        filtered = {name: columns for name, columns in filtered.items() if columns}
                        if filtered:
                            return filtered

    inferred: dict[str, list[str]] = {
        "funding": [],
        "basis": [],
        "volatility": [],
        "liquidity": [],
        "interaction_state": [],
        "other": [],
    }
    for column in feature_columns:
        lowered = column.lower()
        if "funding" in lowered:
            inferred["funding"].append(column)
        elif any(token in lowered for token in ["spread", "basis", "perp_minus_spot"]):
            inferred["basis"].append(column)
        elif any(token in lowered for token in ["vol", "shock", "return"]):
            inferred["volatility"].append(column)
        elif any(token in lowered for token in ["volume", "open_interest", "liquidity"]):
            inferred["liquidity"].append(column)
        elif "x_" in lowered or "_x_" in lowered or "regime" in lowered or "interaction" in lowered:
            inferred["interaction_state"].append(column)
        else:
            inferred["other"].append(column)
    return {name: sorted(columns) for name, columns in inferred.items() if columns}


def _calibration_table(
    predictions: pd.DataFrame,
    *,
    split_name: str,
    bins: int,
) -> pd.DataFrame:
    subset = predictions[predictions["split"] == split_name].copy()
    subset = subset.dropna(subset=["predicted_probability", "actual_label"])
    if subset.empty:
        return pd.DataFrame()
    probabilities = pd.to_numeric(subset["predicted_probability"], errors="coerce")
    if probabilities.nunique(dropna=True) < 2:
        return pd.DataFrame()
    quantile_bins = min(int(bins), int(probabilities.nunique(dropna=True)))
    binned = subset.copy()
    binned["probability_bin"] = pd.qcut(probabilities, q=quantile_bins, duplicates="drop")
    summary = (
        binned.groupby("probability_bin", observed=False)
        .agg(
            row_count=("actual_label", "size"),
            average_predicted_probability=("predicted_probability", "mean"),
            actual_positive_rate=("actual_label", "mean"),
        )
        .reset_index()
    )
    summary["split"] = split_name
    return summary


def _save_frame_if_not_empty(frame: pd.DataFrame, path: Path) -> str | None:
    if frame.empty:
        return None
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return str(path)


def _selected_hyperparameters(settings: DeepLearningSettings) -> dict[str, Any]:
    selected = {
        "lookback_steps": settings.sequence.lookback_steps,
        "learning_rate": settings.training.learning_rate,
        "weight_decay": settings.training.weight_decay,
        "batch_size": settings.training.batch_size,
        "model_name": settings.model.name,
        "dropout": settings.model.dropout,
    }
    if settings.model.name in {"lstm", "gru"}:
        selected.update(
            {
                "hidden_size": settings.model.hidden_size,
                "num_layers": settings.model.num_layers,
                "bidirectional": settings.model.bidirectional,
            }
        )
    elif settings.model.name == "tcn":
        selected.update(
            {
                "tcn_hidden_channels": settings.model.tcn_hidden_channels,
                "tcn_num_blocks": settings.model.tcn_num_blocks,
                "tcn_kernel_size": settings.model.tcn_kernel_size,
                "tcn_dilation_base": settings.model.tcn_dilation_base,
                "tcn_use_residual": settings.model.tcn_use_residual,
            }
        )
    elif settings.model.name == "transformer_encoder":
        selected.update(
            {
                "transformer_d_model": settings.model.transformer_d_model,
                "transformer_nhead": settings.model.transformer_nhead,
                "transformer_num_layers": settings.model.transformer_num_layers,
                "transformer_dim_feedforward": settings.model.transformer_dim_feedforward,
                "transformer_pooling": settings.model.transformer_pooling,
            }
        )
    return selected


def _build_trial_settings(
    settings: DeepLearningSettings,
    candidate: dict[str, Any],
    *,
    epochs_override: int | None = None,
    selection_metric_override: str | None = None,
) -> DeepLearningSettings:
    trial = settings.model_copy(deep=True)
    if "lookback_steps" in candidate:
        trial.sequence.lookback_steps = int(candidate["lookback_steps"])
    if "hidden_size" in candidate:
        trial.model.hidden_size = int(candidate["hidden_size"])
    if "num_layers" in candidate:
        trial.model.num_layers = int(candidate["num_layers"])
    if "dropout" in candidate:
        trial.model.dropout = float(candidate["dropout"])
    if "tcn_hidden_channels" in candidate:
        trial.model.tcn_hidden_channels = int(candidate["tcn_hidden_channels"])
    if "tcn_num_blocks" in candidate:
        trial.model.tcn_num_blocks = int(candidate["tcn_num_blocks"])
    if "tcn_kernel_size" in candidate:
        trial.model.tcn_kernel_size = int(candidate["tcn_kernel_size"])
    if "tcn_dilation_base" in candidate:
        trial.model.tcn_dilation_base = int(candidate["tcn_dilation_base"])
    if "transformer_d_model" in candidate:
        trial.model.transformer_d_model = int(candidate["transformer_d_model"])
    if "transformer_nhead" in candidate:
        trial.model.transformer_nhead = int(candidate["transformer_nhead"])
    if "transformer_num_layers" in candidate:
        trial.model.transformer_num_layers = int(candidate["transformer_num_layers"])
    if "transformer_dim_feedforward" in candidate:
        trial.model.transformer_dim_feedforward = int(candidate["transformer_dim_feedforward"])
    if "transformer_pooling" in candidate:
        trial.model.transformer_pooling = str(candidate["transformer_pooling"])
    if "learning_rate" in candidate:
        trial.training.learning_rate = float(candidate["learning_rate"])
    if "weight_decay" in candidate:
        trial.training.weight_decay = float(candidate["weight_decay"])
    if "batch_size" in candidate:
        trial.training.batch_size = int(candidate["batch_size"])
    if epochs_override is not None:
        trial.training.epochs = int(epochs_override)
    if selection_metric_override is not None:
        trial.training.selection_metric = selection_metric_override
    return trial


def _tuning_candidates(settings: DeepLearningSettings) -> list[dict[str, Any]]:
    tuning = settings.tuning
    candidate_space = {
        "lookback_steps": tuning.lookback_steps or [settings.sequence.lookback_steps],
        "hidden_size": tuning.hidden_size or [settings.model.hidden_size],
        "num_layers": tuning.num_layers or [settings.model.num_layers],
        "dropout": tuning.dropout or [settings.model.dropout],
        "learning_rate": tuning.learning_rate or [settings.training.learning_rate],
        "weight_decay": tuning.weight_decay or [settings.training.weight_decay],
        "batch_size": tuning.batch_size or [settings.training.batch_size],
    }
    keys = list(candidate_space.keys())
    combinations = list(itertools.product(*(candidate_space[key] for key in keys)))
    candidates = [dict(zip(keys, combination, strict=False)) for combination in combinations]
    unique_candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate_json = _json_dumps(candidate)
        if candidate_json in seen:
            continue
        seen.add(candidate_json)
        unique_candidates.append(candidate)
    return unique_candidates[: settings.tuning.max_candidates]


def _time_series_tuning_folds(
    candidate_indices: list[int],
    settings: DeepLearningSettings,
) -> list[tuple[list[int], list[int]]]:
    tuning = settings.tuning
    n_samples = len(candidate_indices)
    if n_samples < tuning.min_train_size + tuning.gap + tuning.n_splits:
        return []
    remaining = n_samples - tuning.min_train_size - tuning.gap
    fold_size = max(1, remaining // tuning.n_splits)
    folds: list[tuple[list[int], list[int]]] = []
    for split_index in range(tuning.n_splits):
        train_end = tuning.min_train_size + split_index * fold_size
        validation_start = train_end + tuning.gap
        if validation_start >= n_samples:
            break
        validation_end = n_samples if split_index == tuning.n_splits - 1 else min(n_samples, validation_start + fold_size)
        if validation_end <= validation_start:
            continue
        train_start = 0
        if tuning.mode == "rolling" and tuning.rolling_window_size is not None:
            train_start = max(0, train_end - tuning.rolling_window_size)
        folds.append(
            (
                candidate_indices[train_start:train_end],
                candidate_indices[validation_start:validation_end],
            )
        )
    return folds


def _fit_model_for_indices(
    frame: pd.DataFrame,
    feature_columns: list[str],
    settings: DeepLearningSettings,
    device: torch.device,
    *,
    train_indices: list[int],
    validation_indices: list[int],
    fit_row_indices: list[int],
    selected_hyperparameters: dict[str, Any],
    feature_importance_method: str,
) -> dict[str, Any]:
    label_diagnostics_by_split = _label_diagnostics_for_frame(frame, settings)
    stats = fit_normalization_stats(
        frame,
        feature_columns,
        settings.target.split_column,
        settings=settings,
        fit_row_indices=fit_row_indices,
    )
    features = transform_features(frame, feature_columns, stats)
    targets = pd.to_numeric(frame[settings.target.column], errors="coerce").astype(float).to_numpy()

    train_dataset = _dataset_from_indices(features, targets, train_indices, settings.sequence.lookback_steps)
    validation_dataset = _dataset_from_indices(features, targets, validation_indices, settings.sequence.lookback_steps)
    pin_memory = device.type == "cuda"
    train_loader = _build_loader(
        train_dataset,
        batch_size=settings.training.batch_size,
        num_workers=settings.training.num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    validation_loader = _build_loader(
        validation_dataset,
        batch_size=settings.training.batch_size,
        num_workers=settings.training.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    model = build_sequence_model(len(feature_columns), settings).to(device)
    train_target_values = targets[np.array(train_indices, dtype=int)]
    loss_fn, loss_metadata = _loss_function(settings, device, train_target_values)
    optimizer = Adam(
        model.parameters(),
        lr=settings.training.learning_rate,
        weight_decay=settings.training.weight_decay,
    )

    best_state_dict: dict[str, Any] | None = None
    best_optimizer_state_dict: dict[str, Any] | None = None
    best_epoch = 0
    best_validation_loss: float | None = None
    best_selection_value: float | None = None
    best_selection_effective_metric = settings.training.selection_metric
    best_selection_effective_value: float | None = None
    best_selection_fallback_used = False
    best_threshold = _default_threshold(settings)
    best_threshold_objective_value: float | None = None
    best_validation_metrics: dict[str, Any] = {}
    best_threshold_search = pd.DataFrame(
        [{"threshold": float(best_threshold), "objective_value": None, "selected": True}]
    )
    history_rows: list[dict[str, Any]] = []
    stale_epochs = 0
    fallback_warning_emitted = False
    selected_hyperparameters_json = _json_dumps(selected_hyperparameters)

    for epoch in range(1, settings.training.epochs + 1):
        train_result = _run_loader(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            clip_grad_norm=settings.training.clip_grad_norm,
        )
        validation_result = _run_loader(
            model=model,
            loader=validation_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=None,
            clip_grad_norm=None,
        )

        validation_score_frame = _score_frame(
            frame,
            validation_result["row_indices"],
            validation_result["outputs"],
            settings,
            prediction_mode="validation_epoch",
            selected_hyperparameters_json=selected_hyperparameters_json,
            threshold_objective=settings.threshold_search.objective if settings.threshold_search.enabled else None,
            feature_importance_method=feature_importance_method,
        )
        threshold_selection = _select_threshold(
            validation_score_frame,
            settings,
            label_diagnostics_by_split=label_diagnostics_by_split,
        )
        epoch_threshold = threshold_selection.selected_threshold
        epoch_threshold_objective_value = threshold_selection.objective_value
        threshold_search = threshold_selection.search_frame
        validation_predictions = _apply_threshold(validation_score_frame, settings, epoch_threshold)
        validation_metrics = _metric_row(validation_predictions, settings)

        train_score_frame = _score_frame(
            frame,
            train_result["row_indices"],
            train_result["outputs"],
            settings,
            prediction_mode="train_epoch",
            selected_hyperparameters_json=selected_hyperparameters_json,
            threshold_objective=settings.threshold_search.objective if settings.threshold_search.enabled else None,
            feature_importance_method=feature_importance_method,
        )
        train_predictions = _apply_threshold(train_score_frame, settings, epoch_threshold)
        train_metrics = _metric_row(train_predictions, settings)

        selection = _selection_resolution(
            settings.training.selection_metric,
            validation_result["loss"],
            validation_metrics,
        )
        if selection["fallback_used"]:
            signal_counts = signal_split_diagnostics(
                validation_predictions,
                signal_column="signal",
                split_names=["validation"],
            )
            diagnostics = DegenerateExperimentDiagnostics(
                stage="checkpoint_selection",
                reason=(
                    f"Checkpoint selection metric '{settings.training.selection_metric}' was undefined on "
                    "validation, so only the fallback metric remained."
                ),
                status="checkpoint_selection_failed",
                model_name=settings.model.name,
                source="deep_learning",
                target_horizon=infer_horizon_label(settings.target.regression_column),
                split="validation",
                selected_threshold=_safe_float(epoch_threshold),
                candidate_threshold_count=summarize_threshold_search(threshold_search).get(
                    "candidate_count"
                ),
                valid_candidate_count=summarize_threshold_search(threshold_search).get(
                    "valid_candidate_count"
                ),
                signal_count_by_split={
                    split_name: int(split_diagnostics.get("signal_count", 0))
                    for split_name, split_diagnostics in signal_counts.items()
                },
                tradeable_rate_by_split=_split_metric_map(
                    label_diagnostics_by_split, "tradeable_rate"
                ),
                profitable_rate_by_split=_split_metric_map(
                    label_diagnostics_by_split, "profitable_rate"
                ),
                future_net_return_bps_by_split=_return_summary_map(
                    label_diagnostics_by_split
                ),
                fallback_used=bool(settings.training.allow_degenerate_fallback),
                fallback_reason=(
                    "Fell back to validation_loss because the configured checkpoint metric was undefined."
                    if settings.training.allow_degenerate_fallback
                    else None
                ),
            )
            if not settings.training.allow_degenerate_fallback:
                raise DegenerateExperimentError(diagnostics)
            if not fallback_warning_emitted:
                warn_on_degenerate_experiment(diagnostics, stacklevel=3)
                fallback_warning_emitted = True
        row = {
            "epoch": epoch,
            "train_loss": _safe_float(train_result["loss"]),
            "validation_loss": _safe_float(validation_result["loss"]),
            "selected_threshold": _safe_float(epoch_threshold),
            "threshold_objective": settings.threshold_search.objective if settings.threshold_search.enabled else None,
            "threshold_objective_value": _safe_float(epoch_threshold_objective_value),
            "selection_metric": settings.training.selection_metric,
            "selection_metric_value": _safe_float(selection["configured_value"]),
            "selection_metric_effective": selection["effective_metric"],
            "selection_metric_effective_value": _safe_float(selection["effective_value"]),
            "selection_metric_fallback_used": bool(selection["fallback_used"]),
        }
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history_rows.append(row)

        if _is_better_metric(
            selection["effective_value"],
            best_selection_effective_value,
            str(selection["effective_metric"]),
        ):
            best_state_dict = copy.deepcopy(model.state_dict())
            best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
            best_epoch = epoch
            best_validation_loss = _safe_float(validation_result["loss"])
            best_selection_value = _safe_float(selection["configured_value"])
            best_selection_effective_metric = str(selection["effective_metric"])
            best_selection_effective_value = _safe_float(selection["effective_value"])
            best_selection_fallback_used = bool(selection["fallback_used"])
            best_threshold = float(epoch_threshold)
            best_threshold_objective_value = _safe_float(epoch_threshold_objective_value)
            best_validation_metrics = dict(validation_metrics)
            best_threshold_search = threshold_search.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= settings.training.early_stopping_patience:
                break

    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())
        best_optimizer_state_dict = copy.deepcopy(optimizer.state_dict())
        best_epoch = settings.training.epochs
        best_validation_loss = None
        best_selection_value = None
        best_selection_effective_metric = settings.training.selection_metric
        best_selection_effective_value = None
        best_selection_fallback_used = False

    model.load_state_dict(best_state_dict)
    return {
        "model": model,
        "stats": stats,
        "history": pd.DataFrame(history_rows),
        "best_epoch": int(best_epoch),
        "best_validation_loss": _safe_float(best_validation_loss),
        "best_selection_value": _safe_float(best_selection_value),
        "best_selection_effective_metric": _safe_text(best_selection_effective_metric),
        "best_selection_effective_value": _safe_float(best_selection_effective_value),
        "best_selection_fallback_used": bool(best_selection_fallback_used),
        "best_threshold": float(best_threshold),
        "best_threshold_objective_value": _safe_float(best_threshold_objective_value),
        "best_validation_metrics": best_validation_metrics,
        "best_threshold_search": best_threshold_search,
        "label_diagnostics_by_split": label_diagnostics_by_split,
        "loss_metadata": loss_metadata,
        "selected_hyperparameters_json": selected_hyperparameters_json,
        "selected_hyperparameters": selected_hyperparameters,
        "optimizer_state_dict": best_optimizer_state_dict,
        "features": features,
    }


def _prediction_frame_from_model(
    model: nn.Module,
    frame: pd.DataFrame,
    features: np.ndarray,
    targets: np.ndarray,
    sample_indices: list[int],
    settings: DeepLearningSettings,
    device: torch.device,
    *,
    threshold: float,
    prediction_mode: str,
    selected_hyperparameters_json: str,
    feature_importance_method: str,
) -> pd.DataFrame:
    if not sample_indices:
        return pd.DataFrame()
    pin_memory = device.type == "cuda"
    dataset = _dataset_from_indices(features, targets, sample_indices, settings.sequence.lookback_steps)
    loader = _build_loader(
        dataset,
        batch_size=settings.training.batch_size,
        num_workers=settings.training.num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    loss_fn, _ = _loss_function(settings, device, targets[np.array(sample_indices, dtype=int)])
    result = _run_loader(
        model=model,
        loader=loader,
        loss_fn=loss_fn,
        device=device,
        optimizer=None,
        clip_grad_norm=None,
    )
    score_frame = _score_frame(
        frame,
        result["row_indices"],
        result["outputs"],
        settings,
        prediction_mode=prediction_mode,
        selected_hyperparameters_json=selected_hyperparameters_json,
        threshold_objective=settings.threshold_search.objective if settings.threshold_search.enabled else None,
        feature_importance_method=feature_importance_method,
    )
    return _apply_threshold(score_frame, settings, threshold)


def _walk_forward_history_indices(
    frame: pd.DataFrame,
    all_sample_indices: list[int],
    settings: DeepLearningSettings,
    *,
    cutoff_timestamp: pd.Timestamp,
    current_split: str,
) -> list[int]:
    split_values = frame[settings.target.split_column].astype(str)
    history = [
        index
        for index in all_sample_indices
        if pd.Timestamp(frame.iloc[index][settings.target.timestamp_column]) < cutoff_timestamp
    ]
    if current_split == "test" and not settings.prediction.use_validation_history_for_test:
        history = [index for index in history if split_values.iloc[index] != "validation"]
    if settings.prediction.mode == "rolling" and settings.prediction.rolling_window_size:
        history = history[-settings.prediction.rolling_window_size :]
    return history


def _walk_forward_predictions(
    frame: pd.DataFrame,
    feature_columns: list[str],
    settings: DeepLearningSettings,
    device: torch.device,
    *,
    base_threshold: float,
    selected_hyperparameters: dict[str, Any],
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    targets = pd.to_numeric(frame[settings.target.column], errors="coerce").astype(float).to_numpy()
    all_indices = build_sequence_indices(frame, settings)
    historical_sample_indices = sorted(all_indices["train"] + all_indices["validation"] + all_indices["test"])
    selected_hyperparameters_json = _json_dumps(selected_hyperparameters)

    static_train_settings = settings.model_copy(deep=True)
    static_train_settings.prediction.mode = "static"
    train_fit = _fit_model_for_indices(
        frame,
        feature_columns,
        static_train_settings,
        device,
        train_indices=all_indices["train"],
        validation_indices=all_indices["validation"],
        fit_row_indices=list(range(0, max(all_indices["train"]) + 1)) if all_indices["train"] else list(range(len(frame))),
        selected_hyperparameters=selected_hyperparameters,
        feature_importance_method="ablation_validation",
    )
    predictions.append(
        _prediction_frame_from_model(
            train_fit["model"],
            frame,
            train_fit["features"],
            targets,
            all_indices["train"],
            settings,
            device,
            threshold=base_threshold,
            prediction_mode="train_fit",
            selected_hyperparameters_json=selected_hyperparameters_json,
            feature_importance_method="ablation_validation",
        )
    )

    for split_name in ["validation", "test"]:
        split_indices = all_indices[split_name]
        if not split_indices:
            continue
        chunk_size = max(1, settings.prediction.refit_every_n_periods)
        for start in range(0, len(split_indices), chunk_size):
            chunk_indices = split_indices[start : start + chunk_size]
            cutoff_timestamp = pd.Timestamp(frame.iloc[chunk_indices[0]][settings.target.timestamp_column])
            history_indices = _walk_forward_history_indices(
                frame,
                historical_sample_indices,
                settings,
                cutoff_timestamp=cutoff_timestamp,
                current_split=split_name,
            )
            if len(history_indices) < settings.prediction.expanding_window_start:
                continue
            train_indices, validation_indices = _history_split_indices(history_indices, settings)
            if not train_indices:
                continue
            fit_start = max(0, train_indices[0] - settings.sequence.lookback_steps + 1)
            fit_end = history_indices[-1]
            fit_row_indices = list(range(fit_start, fit_end + 1))
            refit = _fit_model_for_indices(
                frame,
                feature_columns,
                settings,
                device,
                train_indices=train_indices,
                validation_indices=validation_indices,
                fit_row_indices=fit_row_indices,
                selected_hyperparameters=selected_hyperparameters,
                feature_importance_method="ablation_validation",
            )
            predictions.append(
                _prediction_frame_from_model(
                    refit["model"],
                    frame,
                    refit["features"],
                    targets,
                    chunk_indices,
                    settings,
                    device,
                    threshold=base_threshold,
                    prediction_mode=settings.prediction.mode,
                    selected_hyperparameters_json=selected_hyperparameters_json,
                    feature_importance_method="ablation_validation",
                )
            )
    if not predictions:
        return pd.DataFrame()
    return pd.concat(predictions, ignore_index=True).sort_values(["split", "timestamp"]).reset_index(drop=True)


def _run_feature_group_ablation(
    model: nn.Module,
    frame: pd.DataFrame,
    features: np.ndarray,
    targets: np.ndarray,
    sample_indices: dict[str, list[int]],
    feature_columns: list[str],
    settings: DeepLearningSettings,
    device: torch.device,
    *,
    threshold: float,
    selected_hyperparameters_json: str,
) -> pd.DataFrame:
    if not settings.interpretability.enabled:
        return pd.DataFrame()
    catalog = _feature_group_catalog(settings, feature_columns)
    if settings.interpretability.max_feature_groups is not None:
        catalog = dict(list(catalog.items())[: settings.interpretability.max_feature_groups])
    column_positions = {column: index for index, column in enumerate(feature_columns)}

    rows: list[dict[str, Any]] = []
    for split_name in settings.interpretability.ablation_splits:
        indices = sample_indices.get(split_name, [])
        if not indices:
            continue
        baseline_predictions = _prediction_frame_from_model(
            model,
            frame,
            features,
            targets,
            indices,
            settings,
            device,
            threshold=threshold,
            prediction_mode="static",
            selected_hyperparameters_json=selected_hyperparameters_json,
            feature_importance_method="ablation_validation",
        )
        baseline_metrics = _metric_row(baseline_predictions, settings)
        for group_name, group_columns in catalog.items():
            positions = [column_positions[column] for column in group_columns if column in column_positions]
            if not positions:
                continue
            ablated_features = features.copy()
            ablated_features[:, positions] = 0.0
            predictions = _prediction_frame_from_model(
                model,
                frame,
                ablated_features,
                targets,
                indices,
                settings,
                device,
                threshold=threshold,
                prediction_mode="static",
                selected_hyperparameters_json=selected_hyperparameters_json,
                feature_importance_method="ablation_validation",
            )
            ablated_metrics = _metric_row(predictions, settings)
            baseline_avg = _safe_float(baseline_metrics.get("avg_signal_return_bps"))
            ablated_avg = _safe_float(ablated_metrics.get("avg_signal_return_bps"))
            rows.append(
                {
                    "split": split_name,
                    "feature_group": group_name,
                    "feature_count": len(group_columns),
                    "baseline_avg_signal_return_bps": baseline_avg,
                    "ablated_avg_signal_return_bps": ablated_avg,
                    "delta_avg_signal_return_bps": (
                        float(ablated_avg - baseline_avg)
                        if baseline_avg is not None and ablated_avg is not None
                        else None
                    ),
                    "baseline_signal_hit_rate": baseline_metrics.get("signal_hit_rate"),
                    "ablated_signal_hit_rate": ablated_metrics.get("signal_hit_rate"),
                    "baseline_cumulative_signal_return_bps": baseline_metrics.get("cumulative_signal_return_bps"),
                    "ablated_cumulative_signal_return_bps": ablated_metrics.get("cumulative_signal_return_bps"),
                }
            )
    return pd.DataFrame(rows)


def _leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()
    leaderboard = metrics[metrics["split"].isin(["validation", "test"])].copy()
    leaderboard["has_signals"] = leaderboard["signal_count"].fillna(0).astype(float) > 0.0
    sort_keys = [
        "has_signals",
        "task",
        "split",
        "avg_signal_return_bps",
        "cumulative_signal_return_bps",
        "signal_hit_rate",
        "pearson_corr",
        "f1",
        "roc_auc",
    ]
    available_keys = [key for key in sort_keys if key in leaderboard.columns]
    if available_keys:
        leaderboard = leaderboard.sort_values(available_keys, ascending=[False] * len(available_keys), na_position="last")
    return leaderboard.reset_index(drop=True)


def _write_frame(frame: pd.DataFrame, path: Path) -> str:
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")
    return str(path)


def _output_dir(settings: DeepLearningSettings) -> Path:
    return ensure_directory(
        _resolve_path(settings.output.model_dir)
        / settings.input.provider
        / settings.input.symbol.lower()
        / settings.input.frequency
        / settings.output.run_name
    )


def _tune_hyperparameters(
    frame: pd.DataFrame,
    feature_columns: list[str],
    settings: DeepLearningSettings,
    device: torch.device,
) -> tuple[DeepLearningSettings, pd.DataFrame]:
    if not settings.tuning.enabled:
        return settings, pd.DataFrame()
    rows: list[dict[str, Any]] = []
    best_candidate_settings = settings
    best_score: float | None = None
    metric_name = settings.tuning.metric
    direction = SELECTION_METRIC_DIRECTIONS.get(metric_name, "max")

    for candidate in _tuning_candidates(settings):
        candidate_settings = _build_trial_settings(
            settings,
            candidate,
            epochs_override=settings.tuning.trial_epochs,
            selection_metric_override=settings.tuning.metric,
        )
        candidate_indices = build_sequence_indices(frame, candidate_settings)["train"]
        folds = _time_series_tuning_folds(candidate_indices, candidate_settings)
        if not folds:
            continue
        fold_scores: list[float] = []
        for fold_index, (train_indices, validation_indices) in enumerate(folds, start=1):
            fit_start = max(0, train_indices[0] - candidate_settings.sequence.lookback_steps + 1)
            fit_end = train_indices[-1]
            fit_row_indices = list(range(fit_start, fit_end + 1))
            result = _fit_model_for_indices(
                frame,
                feature_columns,
                candidate_settings,
                device,
                train_indices=train_indices,
                validation_indices=validation_indices,
                fit_row_indices=fit_row_indices,
                selected_hyperparameters=candidate,
                feature_importance_method="ablation_validation",
            )
            score = _safe_float(result["best_selection_effective_value"])
            if score is not None:
                fold_scores.append(float(score))
            rows.append(
                {
                    "candidate_json": _json_dumps(candidate),
                    "fold": fold_index,
                    "selection_metric": metric_name,
                    "selection_value": score,
                    "selection_metric_effective": result["best_selection_effective_metric"],
                    "selection_metric_fallback_used": bool(result["best_selection_fallback_used"]),
                    "best_epoch": result["best_epoch"],
                    "selected_threshold": result["best_threshold"],
                }
            )
        if not fold_scores:
            continue
        mean_score = float(np.mean(fold_scores))
        if best_score is None or (direction == "min" and mean_score < best_score) or (direction != "min" and mean_score > best_score):
            best_score = mean_score
            best_candidate_settings = _build_trial_settings(settings, candidate)
    return best_candidate_settings, pd.DataFrame(rows)


def _write_report(
    settings: DeepLearningSettings,
    output_dir: Path,
    feature_columns: list[str],
    history: pd.DataFrame,
    metrics: pd.DataFrame,
    leaderboard: pd.DataFrame,
    best_epoch: int,
    device: str,
    sample_counts: dict[str, int],
    *,
    selected_threshold: float,
    best_selection_value: float | None,
    best_selection_effective_metric: str | None,
    best_selection_effective_value: float | None,
    best_selection_fallback_used: bool,
    loss_metadata: dict[str, Any],
    tuning_results: pd.DataFrame,
    diagnostic_paths: dict[str, str],
    label_diagnostics_by_split: dict[str, dict[str, Any]],
    signal_count_by_split: dict[str, int],
    degenerate_experiment: bool,
    degenerate_stage: str | None,
    degenerate_reason: str | None,
    fallback_used: bool,
    fallback_reason: str | None,
    threshold_search_summary: dict[str, Any],
) -> str | None:
    if not settings.output.write_markdown_report:
        return None
    report_path = output_dir / "training_report.md"
    lines = [
        "# Deep Learning Experiment Report",
        "",
        f"- Model: `{settings.model.name}`",
        f"- Task: `{settings.target.task}`",
        f"- Target column: `{settings.target.column}`",
        f"- Lookback steps: `{settings.sequence.lookback_steps}`",
        f"- Feature count: `{len(feature_columns)}`",
        f"- Device: `{device}`",
        f"- Best epoch: `{best_epoch}`",
        f"- Checkpoint selection metric: `{settings.training.selection_metric}`",
        f"- Best checkpoint selection value: `{best_selection_value}`",
        f"- Effective checkpoint selection metric: `{best_selection_effective_metric}`",
        f"- Effective checkpoint selection value: `{best_selection_effective_value}`",
        f"- Checkpoint metric fallback used: `{best_selection_fallback_used}`",
        f"- Degenerate experiment: `{degenerate_experiment}`",
        f"- Degenerate stage: `{degenerate_stage}`",
        f"- Degenerate reason: `{degenerate_reason}`",
        f"- Fallback used: `{fallback_used}`",
        f"- Fallback reason: `{fallback_reason}`",
        f"- Selected threshold: `{selected_threshold}`",
        f"- Threshold objective: `{settings.threshold_search.objective if settings.threshold_search.enabled else 'disabled'}`",
        f"- Threshold search summary: `{threshold_search_summary}`",
        f"- Signal count by split: `{signal_count_by_split}`",
        f"- Tradeable rate by split: `{_split_metric_map(label_diagnostics_by_split, 'tradeable_rate')}`",
        f"- Profitable rate by split: `{_split_metric_map(label_diagnostics_by_split, 'profitable_rate')}`",
        f"- Prediction mode: `{settings.prediction.mode}`",
        f"- Loss metadata: `{loss_metadata}`",
        f"- Preprocessing scaler: `{settings.preprocessing.scaler}`",
        f"- Winsorization: `({settings.preprocessing.winsorize_lower_quantile}, {settings.preprocessing.winsorize_upper_quantile})`",
        f"- Tuning enabled: `{settings.tuning.enabled}`",
        f"- Interpretability enabled: `{settings.interpretability.enabled}`",
        f"- Sample counts: `{sample_counts}`",
        "",
        "## Validation/Test Metrics",
        "",
        _table_to_markdown(leaderboard),
        "",
        "## Full Metrics",
        "",
        _table_to_markdown(metrics),
        "",
        "## Label Diagnostics",
        "",
        _table_to_markdown(pd.DataFrame(label_diagnostics_by_split).T.reset_index().rename(columns={"index": "split"})),
        "",
        "## Training History",
        "",
        _table_to_markdown(history),
    ]
    if not tuning_results.empty:
        lines.extend(["", "## Tuning Results", "", _table_to_markdown(tuning_results)])
    if diagnostic_paths:
        lines.extend(
            [
                "",
                "## Diagnostics",
                "",
                "\n".join(f"- `{name}`: `{path}`" for name, path in sorted(diagnostic_paths.items())),
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Sequences use only the target row and preceding feature rows; future context is never included.",
            "- Validation and test rows can reuse earlier historical rows as context when allow_cross_split_context=true.",
            "- Threshold search is validation-based and can optimize trading-oriented signal metrics rather than raw loss only.",
            "- Walk-forward prediction mode periodically re-fits on historical data only; the saved checkpoint still refers to the base train/validation model.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def run_deep_learning_pipeline(settings: DeepLearningSettings) -> DeepLearningArtifacts:
    """Train the sequence-model experiment and save reproducible artifacts."""
    _set_random_seed(settings.training.seed, settings.training.deterministic)
    base_frame = _load_supervised_dataset(settings)
    feature_columns = select_feature_columns(base_frame, settings)
    device = _select_device(settings.training.device)

    effective_settings, tuning_results = _tune_hyperparameters(base_frame, feature_columns, settings, device)
    frame = base_frame.copy()
    sample_indices = build_sequence_indices(frame, effective_settings)
    if not sample_indices["train"]:
        raise ValueError("No train sequences were generated for the configured deep-learning task.")
    if not sample_indices["validation"]:
        raise ValueError("No validation sequences were generated for the configured deep-learning task.")

    fit_row_indices = list(range(0, max(sample_indices["train"]) + 1))
    selected_hyperparameters = _selected_hyperparameters(effective_settings)
    fit_result = _fit_model_for_indices(
        frame,
        feature_columns,
        effective_settings,
        device,
        train_indices=sample_indices["train"],
        validation_indices=sample_indices["validation"],
        fit_row_indices=fit_row_indices,
        selected_hyperparameters=selected_hyperparameters,
        feature_importance_method="ablation_validation",
    )

    output_dir = _output_dir(effective_settings)
    diagnostics_dir = ensure_directory(output_dir / "diagnostics")
    checkpoint_path = output_dir / "best_model.pt"
    feature_columns_path = output_dir / "feature_columns.json"
    normalization_path = output_dir / "feature_normalization.json"
    feature_columns_path.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")
    normalization_path.write_text(
        json.dumps(
            {
                "medians": fit_result["stats"].medians,
                "centers": fit_result["stats"].centers,
                "scales": fit_result["stats"].scales,
                "lower_bounds": fit_result["stats"].lower_bounds,
                "upper_bounds": fit_result["stats"].upper_bounds,
                "scaler": fit_result["stats"].scaler,
                "winsorize_lower_quantile": fit_result["stats"].winsorize_lower_quantile,
                "winsorize_upper_quantile": fit_result["stats"].winsorize_upper_quantile,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    torch.save(
        {
            "epoch": fit_result["best_epoch"],
            "model_state_dict": fit_result["model"].state_dict(),
            "optimizer_state_dict": fit_result["optimizer_state_dict"],
            "config": effective_settings.model_dump(),
            "feature_columns": feature_columns,
            "normalization": {
                "medians": fit_result["stats"].medians,
                "centers": fit_result["stats"].centers,
                "scales": fit_result["stats"].scales,
                "lower_bounds": fit_result["stats"].lower_bounds,
                "upper_bounds": fit_result["stats"].upper_bounds,
                "scaler": fit_result["stats"].scaler,
                "winsorize_lower_quantile": fit_result["stats"].winsorize_lower_quantile,
                "winsorize_upper_quantile": fit_result["stats"].winsorize_upper_quantile,
            },
            "best_validation_loss": fit_result["best_validation_loss"],
            "best_selection_metric": effective_settings.training.selection_metric,
            "best_selection_value": fit_result["best_selection_value"],
            "best_selection_effective_metric": fit_result["best_selection_effective_metric"],
            "best_selection_effective_value": fit_result["best_selection_effective_value"],
            "best_selection_fallback_used": fit_result["best_selection_fallback_used"],
            "selected_threshold": fit_result["best_threshold"],
            "selected_threshold_objective_value": fit_result["best_threshold_objective_value"],
            "selected_hyperparameters": selected_hyperparameters,
        },
        checkpoint_path,
    )

    targets = pd.to_numeric(frame[effective_settings.target.column], errors="coerce").astype(float).to_numpy()
    selected_hyperparameters_json = fit_result["selected_hyperparameters_json"]
    diagnostic_paths: dict[str, str] = {}

    threshold_search_path = diagnostics_dir / "threshold_search.csv"
    fit_result["best_threshold_search"].to_csv(threshold_search_path, index=False)
    diagnostic_paths["threshold_search"] = str(threshold_search_path)

    tuning_results_path = diagnostics_dir / "tuning_results.csv"
    if not tuning_results.empty:
        tuning_results.to_csv(tuning_results_path, index=False)
        diagnostic_paths["tuning_results"] = str(tuning_results_path)

    if effective_settings.prediction.mode == "static":
        predictions_by_split = [
            _prediction_frame_from_model(
                fit_result["model"],
                frame,
                fit_result["features"],
                targets,
                sample_indices[split_name],
                effective_settings,
                device,
                threshold=fit_result["best_threshold"],
                prediction_mode="static" if split_name != "train" else "train_fit",
                selected_hyperparameters_json=selected_hyperparameters_json,
                feature_importance_method="ablation_validation",
            )
            for split_name in ["train", "validation", "test"]
            if sample_indices[split_name]
        ]
        predictions = pd.concat(predictions_by_split, ignore_index=True).sort_values(["split", "timestamp"]).reset_index(drop=True)
    else:
        predictions = _walk_forward_predictions(
            frame,
            feature_columns,
            effective_settings,
            device,
            base_threshold=fit_result["best_threshold"],
            selected_hyperparameters=selected_hyperparameters,
        )

    prediction_metadata_columns = {
        "selected_threshold": fit_result["best_threshold"],
        "selected_threshold_objective": effective_settings.threshold_search.objective if effective_settings.threshold_search.enabled else None,
        "selected_threshold_objective_value": fit_result["best_threshold_objective_value"],
        "threshold_objective": effective_settings.threshold_search.objective if effective_settings.threshold_search.enabled else None,
        "checkpoint_selection_metric": effective_settings.training.selection_metric,
        "best_checkpoint_metric_value": fit_result["best_selection_value"],
        "checkpoint_selection_effective_metric": fit_result["best_selection_effective_metric"],
        "best_checkpoint_effective_metric_value": fit_result["best_selection_effective_value"],
        "checkpoint_selection_fallback_used": fit_result["best_selection_fallback_used"],
        "selected_loss": fit_result["loss_metadata"].get("loss_name"),
        "prediction_mode": effective_settings.prediction.mode,
        "regression_loss": effective_settings.training.regression_loss if effective_settings.target.task == "regression" else None,
        "use_balanced_classification_loss": effective_settings.training.use_balanced_classification_loss if effective_settings.target.task == "classification" else None,
        "preprocessing_scaler": effective_settings.preprocessing.scaler,
        "winsorize_lower_quantile": effective_settings.preprocessing.winsorize_lower_quantile,
        "winsorize_upper_quantile": effective_settings.preprocessing.winsorize_upper_quantile,
    }
    for column, value in prediction_metadata_columns.items():
        predictions[column] = value

    metrics = evaluate_prediction_table(
        predictions,
        top_quantile=effective_settings.threshold_search.top_quantile,
    )
    signal_diagnostics = signal_split_diagnostics(predictions, signal_column="signal")
    signal_count_by_split = {
        split_name: int(split_diagnostics.get("signal_count", 0))
        for split_name, split_diagnostics in signal_diagnostics.items()
    }
    signal_warning = _warn_on_zero_signal_splits(
        settings=effective_settings,
        label_diagnostics_by_split=fit_result["label_diagnostics_by_split"],
        predictions=predictions,
    )
    threshold_search_summary = summarize_threshold_search(fit_result["best_threshold_search"])
    threshold_fallback_used = bool(
        fit_result["best_threshold_search"].get("fallback_used", pd.Series(dtype=bool))
        .fillna(False)
        .astype(bool)
        .any()
    )
    threshold_fallback_reason_series = fit_result["best_threshold_search"].get(
        "fallback_reason",
        pd.Series(dtype=object),
    )
    threshold_fallback_reason = None
    if hasattr(threshold_fallback_reason_series, "dropna"):
        non_null_reasons = threshold_fallback_reason_series.dropna().astype(str)
        if not non_null_reasons.empty:
            threshold_fallback_reason = non_null_reasons.iloc[0]
    checkpoint_fallback_reason = (
        "Fell back to validation_loss because the configured checkpoint metric was undefined."
        if fit_result["best_selection_fallback_used"]
        else None
    )
    fallback_used = bool(fit_result["best_selection_fallback_used"] or threshold_fallback_used)
    fallback_reason = threshold_fallback_reason or checkpoint_fallback_reason
    degenerate_experiment = bool(
        signal_warning is not None or fallback_used
    )
    degenerate_stage = (
        signal_warning.stage
        if signal_warning is not None
        else "threshold_search"
        if threshold_fallback_used
        else "checkpoint_selection"
        if fit_result["best_selection_fallback_used"]
        else None
    )
    degenerate_reason = (
        signal_warning.reason
        if signal_warning is not None
        else fallback_reason
    )
    metadata_columns = {
        **prediction_metadata_columns,
        "threshold_objective": effective_settings.threshold_search.objective if effective_settings.threshold_search.enabled else None,
        "degenerate_experiment": degenerate_experiment,
        "degenerate_stage": degenerate_stage,
        "degenerate_reason": degenerate_reason,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
    }
    for column, value in metadata_columns.items():
        metrics[column] = value
    leaderboard = _leaderboard(metrics)

    history = fit_result["history"]
    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)

    predictions_path = output_dir / "dl_predictions.parquet"
    metrics_path = output_dir / "dl_metrics.parquet"
    predictions_primary_path = _write_frame(predictions, predictions_path)
    metrics_primary_path = _write_frame(metrics, metrics_path)

    predictions_csv_path: str | None = None
    metrics_csv_path: str | None = None
    leaderboard_csv_path: str | None = None
    if effective_settings.output.write_csv:
        predictions_csv_path = _write_frame(predictions, predictions_path.with_suffix(".csv"))
        metrics_csv_path = _write_frame(metrics, metrics_path.with_suffix(".csv"))
        leaderboard_csv_path = _write_frame(leaderboard, output_dir / "dl_leaderboard.csv")
    leaderboard_path = _write_frame(leaderboard, output_dir / "dl_leaderboard.parquet")

    if effective_settings.target.task == "classification":
        for split_name in ["validation", "test"]:
            calibration = _calibration_table(
                predictions,
                split_name=split_name,
                bins=effective_settings.interpretability.calibration_bins,
            )
            calibration_path = _save_frame_if_not_empty(
                calibration,
                diagnostics_dir / f"calibration_{split_name}.csv",
            )
            if calibration_path is not None:
                diagnostic_paths[f"calibration_{split_name}"] = calibration_path

    ablation = _run_feature_group_ablation(
        fit_result["model"],
        frame,
        fit_result["features"],
        targets,
        sample_indices,
        feature_columns,
        effective_settings,
        device,
        threshold=fit_result["best_threshold"],
        selected_hyperparameters_json=selected_hyperparameters_json,
    )
    ablation_path = _save_frame_if_not_empty(ablation, diagnostics_dir / "feature_group_ablation.csv")
    if ablation_path is not None:
        diagnostic_paths["feature_group_ablation"] = ablation_path

    report_path = _write_report(
        settings=effective_settings,
        output_dir=output_dir,
        feature_columns=feature_columns,
        history=history,
        metrics=metrics,
        leaderboard=leaderboard,
        best_epoch=fit_result["best_epoch"],
        device=str(device),
        sample_counts={split_name: len(indices) for split_name, indices in sample_indices.items()},
        selected_threshold=fit_result["best_threshold"],
        best_selection_value=fit_result["best_selection_value"],
        best_selection_effective_metric=fit_result["best_selection_effective_metric"],
        best_selection_effective_value=fit_result["best_selection_effective_value"],
        best_selection_fallback_used=fit_result["best_selection_fallback_used"],
        loss_metadata=fit_result["loss_metadata"],
        tuning_results=tuning_results,
        diagnostic_paths=diagnostic_paths,
        label_diagnostics_by_split=fit_result["label_diagnostics_by_split"],
        signal_count_by_split=signal_count_by_split,
        degenerate_experiment=degenerate_experiment,
        degenerate_stage=degenerate_stage,
        degenerate_reason=degenerate_reason,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        threshold_search_summary=threshold_search_summary,
    )

    manifest = {
        "input": effective_settings.input.model_dump(),
        "target": effective_settings.target.model_dump(),
        "feature_selection": effective_settings.feature_selection.model_dump(),
        "sequence": effective_settings.sequence.model_dump(),
        "model": effective_settings.model.model_dump(),
        "preprocessing": effective_settings.preprocessing.model_dump(),
        "threshold_search": effective_settings.threshold_search.model_dump(),
        "tuning": effective_settings.tuning.model_dump(),
        "prediction": effective_settings.prediction.model_dump(),
        "interpretability": effective_settings.interpretability.model_dump(),
        "training": effective_settings.training.model_dump(),
        "row_count": int(len(frame)),
        "feature_count": len(feature_columns),
        "sample_counts": {split_name: len(indices) for split_name, indices in sample_indices.items()},
        "signal_count_by_split": signal_count_by_split,
        "label_diagnostics_by_split": fit_result["label_diagnostics_by_split"],
        "tradeable_rate_by_split": _split_metric_map(
            fit_result["label_diagnostics_by_split"], "tradeable_rate"
        ),
        "profitable_rate_by_split": _split_metric_map(
            fit_result["label_diagnostics_by_split"], "profitable_rate"
        ),
        "degenerate_experiment": degenerate_experiment,
        "status": "ok" if not degenerate_experiment else "warning",
        "degenerate_stage": degenerate_stage,
        "degenerate_reason": degenerate_reason,
        "reason": degenerate_reason,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "best_epoch": int(fit_result["best_epoch"]),
        "best_validation_loss": _safe_float(fit_result["best_validation_loss"]),
        "best_checkpoint_metric": effective_settings.training.selection_metric,
        "best_checkpoint_metric_value": _safe_float(fit_result["best_selection_value"]),
        "best_checkpoint_effective_metric": _safe_text(fit_result["best_selection_effective_metric"]),
        "best_checkpoint_effective_metric_value": _safe_float(fit_result["best_selection_effective_value"]),
        "checkpoint_metric_fallback_used": bool(fit_result["best_selection_fallback_used"]),
        "selected_threshold": _safe_float(fit_result["best_threshold"]),
        "selected_threshold_objective": effective_settings.threshold_search.objective if effective_settings.threshold_search.enabled else None,
        "selected_threshold_objective_value": _safe_float(fit_result["best_threshold_objective_value"]),
        "threshold_search_summary": threshold_search_summary,
        "selected_loss": fit_result["loss_metadata"].get("loss_name"),
        "loss_metadata": fit_result["loss_metadata"],
        "selected_hyperparameters": selected_hyperparameters,
        "prediction_mode": effective_settings.prediction.mode,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "predictions_path": predictions_primary_path,
        "predictions_csv_path": predictions_csv_path,
        "metrics_path": metrics_primary_path,
        "metrics_csv_path": metrics_csv_path,
        "leaderboard_path": leaderboard_path,
        "leaderboard_csv_path": leaderboard_csv_path,
        "feature_columns_path": str(feature_columns_path),
        "normalization_path": str(normalization_path),
        "report_path": report_path,
        "diagnostic_paths": diagnostic_paths,
        "tuning_results_path": diagnostic_paths.get("tuning_results"),
    }
    manifest_path = output_dir / "dl_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return DeepLearningArtifacts(
        output_dir=str(output_dir),
        checkpoint_path=str(checkpoint_path),
        history_path=str(history_path),
        predictions_path=predictions_primary_path,
        predictions_csv_path=predictions_csv_path,
        metrics_path=metrics_primary_path,
        metrics_csv_path=metrics_csv_path,
        leaderboard_path=leaderboard_path,
        leaderboard_csv_path=leaderboard_csv_path,
        report_path=report_path,
        manifest_path=str(manifest_path),
        feature_columns_path=str(feature_columns_path),
        normalization_path=str(normalization_path),
        diagnostic_paths=diagnostic_paths,
    )
