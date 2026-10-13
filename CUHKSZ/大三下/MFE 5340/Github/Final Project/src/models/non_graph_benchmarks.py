"""Non-graph benchmark models for Stage 3.

The implementations are intentionally transparent rather than exact paper
replications. They provide characteristic-only baselines before graph context is
introduced in later stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.torch_models import ConditionalBetaMLP, MLPReturnPredictor, TorchModelConfig
from src.training.train import get_torch_device, set_global_seed


@dataclass(frozen=True)
class NeuralTrainConfig:
    """Shared torch training hyperparameters."""

    hidden_dim: int = 64
    dropout: float = 0.1
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    batch_size: int = 4096
    max_epochs: int = 20
    patience: int = 4
    seed: int = 20260408
    device: str = "auto"


def _device_from_config(device: str) -> torch.device:
    if device == "auto":
        return get_torch_device(prefer_cuda=True)
    return torch.device(device)


def _array(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    return frame[feature_cols].to_numpy(dtype=np.float32, copy=True)


def _target(frame: pd.DataFrame, target_col: str) -> np.ndarray:
    return frame[target_col].to_numpy(dtype=np.float32, copy=True)


def _make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, extra: np.ndarray | None = None) -> DataLoader:
    tensors: list[Tensor] = [torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)]
    if extra is not None:
        tensors.append(torch.as_tensor(extra, dtype=torch.long))
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class MLPBenchmark:
    """Direct characteristic-only next-month return predictor."""

    model_name = "mlp_predictor"

    def __init__(self, feature_cols: list[str], target_col: str, config: NeuralTrainConfig) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.config = config
        self.device = _device_from_config(config.device)
        self.model: MLPReturnPredictor | None = None
        self.best_validation_loss: float | None = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "MLPBenchmark":
        set_global_seed(self.config.seed)
        model_config = TorchModelConfig(
            input_dim=len(self.feature_cols),
            hidden_dim=self.config.hidden_dim,
            output_dim=1,
            dropout=self.config.dropout,
        )
        model = MLPReturnPredictor(model_config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        loss_fn = nn.MSELoss()

        train_loader = _make_loader(_array(train, self.feature_cols), _target(train, self.target_col), self.config.batch_size, shuffle=True)
        val_x = torch.as_tensor(_array(validation, self.feature_cols), dtype=torch.float32, device=self.device)
        val_y = torch.as_tensor(_target(validation, self.target_col), dtype=torch.float32, device=self.device)

        best_state: dict[str, Tensor] | None = None
        best_loss = np.inf
        bad_epochs = 0
        for _epoch in range(self.config.max_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
            if val_loss < best_loss - 1.0e-8:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.config.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self.model = model
        self.best_validation_loss = best_loss
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        x = torch.as_tensor(_array(frame, self.feature_cols), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x).detach().cpu().numpy()
        return pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "stock_id": frame["stock_id"].to_numpy(),
                "model": self.model_name,
                "y_true": frame[self.target_col].to_numpy(dtype=np.float64),
                "y_pred": pred.astype(np.float64),
            }
        )


class IPCAStyleBenchmark:
    """Linear characteristic-driven latent factor benchmark.

    The model minimizes r_{i,t+1} ~= (x_{i,t}' Gamma) f_t with alternating least
    squares. OOS forecasts use the historical mean factor premium from the
    training window.
    """

    model_name = "ipca_style"

    def __init__(
        self,
        feature_cols: list[str],
        target_col: str,
        n_factors: int,
        ridge_alpha: float = 1.0e-3,
        als_iterations: int = 3,
        seed: int = 20260408,
    ) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_factors = n_factors
        self.ridge_alpha = ridge_alpha
        self.als_iterations = als_iterations
        self.seed = seed
        self.gamma_: np.ndarray | None = None
        self.factor_by_date_: pd.DataFrame | None = None
        self.factor_mean_: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, _validation: pd.DataFrame | None = None) -> "IPCAStyleBenchmark":
        rng = np.random.default_rng(self.seed)
        frame = train.sort_values(["date", "stock_id"]).reset_index(drop=True)
        x = frame[self.feature_cols].to_numpy(dtype=np.float64, copy=True)
        y = frame[self.target_col].to_numpy(dtype=np.float64, copy=True)
        dates = pd.to_datetime(frame["date"])
        unique_dates = pd.DatetimeIndex(dates.drop_duplicates()).sort_values()
        date_lookup = {date: idx for idx, date in enumerate(unique_dates)}
        date_codes = dates.map(date_lookup).to_numpy(dtype=np.int64)
        p = x.shape[1]
        k = self.n_factors
        gamma = rng.normal(scale=0.01, size=(p, k))

        factors = pd.DataFrame(index=unique_dates, columns=[f"factor_{j + 1}" for j in range(k)], dtype="float64")
        for _iteration in range(self.als_iterations):
            factor_lookup: dict[int, np.ndarray] = {}
            for date_idx, date in enumerate(unique_dates):
                mask = date_codes == date_idx
                beta_t = x[mask] @ gamma
                lhs = beta_t.T @ beta_t + self.ridge_alpha * np.eye(k)
                rhs = beta_t.T @ y[mask]
                factor = np.linalg.solve(lhs, rhs)
                factor_lookup[date_idx] = factor
                factors.loc[date] = factor

            f_rows = np.vstack([factor_lookup[int(code)] for code in date_codes])
            design = (x[:, :, None] * f_rows[:, None, :]).reshape(x.shape[0], p * k)
            ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False, solver="lsqr")
            ridge.fit(design, y)
            gamma = ridge.coef_.reshape(p, k)

        self.gamma_ = gamma
        self.factor_by_date_ = factors.astype(float)
        self.factor_mean_ = self.factor_by_date_.mean(axis=0).to_numpy(dtype=np.float64)
        return self

    def _beta(self, frame: pd.DataFrame) -> np.ndarray:
        if self.gamma_ is None:
            raise RuntimeError("model is not fitted")
        return frame[self.feature_cols].to_numpy(dtype=np.float64, copy=True) @ self.gamma_

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.factor_mean_ is None:
            raise RuntimeError("model is not fitted")
        beta = self._beta(frame)
        pred = beta @ self.factor_mean_
        return pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "stock_id": frame["stock_id"].to_numpy(),
                "model": self.model_name,
                "y_true": frame[self.target_col].to_numpy(dtype=np.float64),
                "y_pred": pred.astype(np.float64),
            }
        )

    def exposures(self, frame: pd.DataFrame) -> pd.DataFrame:
        beta = self._beta(frame)
        out = frame[["date", "stock_id"]].copy()
        out["model"] = self.model_name
        for j in range(beta.shape[1]):
            out[f"beta_{j + 1}"] = beta[:, j]
        return out

    def latent_factors(self) -> pd.DataFrame:
        if self.factor_by_date_ is None or self.factor_mean_ is None:
            raise RuntimeError("model is not fitted")
        factors = self.factor_by_date_.reset_index().rename(columns={"index": "date"})
        factors["model"] = self.model_name
        factors["factor_kind"] = "train_factor"
        mean_row: dict[str, Any] = {"date": pd.NaT, "model": self.model_name, "factor_kind": "forecast_mean"}
        for j, value in enumerate(self.factor_mean_, start=1):
            mean_row[f"factor_{j}"] = value
        return pd.concat([factors, pd.DataFrame([mean_row])], ignore_index=True)


class CAEStyleBenchmark:
    """Nonlinear characteristic-driven conditional autoencoder-style benchmark."""

    model_name = "conditional_autoencoder_style"

    def __init__(self, feature_cols: list[str], target_col: str, n_factors: int, config: NeuralTrainConfig) -> None:
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.n_factors = n_factors
        self.config = config
        self.device = _device_from_config(config.device)
        self.model: _CAEModel | None = None
        self.train_dates_: pd.DatetimeIndex | None = None
        self.factor_mean_: np.ndarray | None = None
        self.best_validation_loss: float | None = None

    def fit(self, train: pd.DataFrame, validation: pd.DataFrame) -> "CAEStyleBenchmark":
        set_global_seed(self.config.seed)
        train_frame = train.sort_values(["date", "stock_id"]).reset_index(drop=True)
        train_dates = pd.DatetimeIndex(pd.to_datetime(train_frame["date"].unique())).sort_values()
        date_lookup = {date: idx for idx, date in enumerate(train_dates)}
        date_idx = pd.to_datetime(train_frame["date"]).map(date_lookup).to_numpy(dtype=np.int64)

        model_config = TorchModelConfig(
            input_dim=len(self.feature_cols),
            hidden_dim=self.config.hidden_dim,
            n_factors=self.n_factors,
            dropout=self.config.dropout,
        )
        model = _CAEModel(model_config, n_dates=len(train_dates)).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        loss_fn = nn.MSELoss()

        train_loader = _make_loader(
            _array(train_frame, self.feature_cols),
            _target(train_frame, self.target_col),
            self.config.batch_size,
            shuffle=True,
            extra=date_idx,
        )
        val_x = torch.as_tensor(_array(validation, self.feature_cols), dtype=torch.float32, device=self.device)
        val_y = torch.as_tensor(_target(validation, self.target_col), dtype=torch.float32, device=self.device)

        best_state: dict[str, Tensor] | None = None
        best_loss = np.inf
        bad_epochs = 0
        for _epoch in range(self.config.max_epochs):
            model.train()
            for batch_x, batch_y, batch_date_idx in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_date_idx = batch_date_idx.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                pred, _beta = model(batch_x, batch_date_idx)
                loss = loss_fn(pred, batch_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred, _ = model.predict_with_factor_mean(val_x)
                val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
            if val_loss < best_loss - 1.0e-8:
                best_loss = val_loss
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= self.config.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        with torch.no_grad():
            factor_mean = model.factor_mean().detach().cpu().numpy().astype(np.float64)
        self.model = model
        self.train_dates_ = train_dates
        self.factor_mean_ = factor_mean
        self.best_validation_loss = best_loss
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        x = torch.as_tensor(_array(frame, self.feature_cols), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            pred, _beta = self.model.predict_with_factor_mean(x)
        return pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "stock_id": frame["stock_id"].to_numpy(),
                "model": self.model_name,
                "y_true": frame[self.target_col].to_numpy(dtype=np.float64),
                "y_pred": pred.detach().cpu().numpy().astype(np.float64),
            }
        )

    def exposures(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        x = torch.as_tensor(_array(frame, self.feature_cols), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            _pred, beta = self.model.predict_with_factor_mean(x)
        beta_np = beta.detach().cpu().numpy().astype(np.float64)
        out = frame[["date", "stock_id"]].copy()
        out["model"] = self.model_name
        for j in range(beta_np.shape[1]):
            out[f"beta_{j + 1}"] = beta_np[:, j]
        return out

    def latent_factors(self) -> pd.DataFrame:
        if self.model is None or self.train_dates_ is None or self.factor_mean_ is None:
            raise RuntimeError("model is not fitted")
        with torch.no_grad():
            factors_np = self.model.factor_embeddings.weight.detach().cpu().numpy().astype(np.float64)
        factors = pd.DataFrame(factors_np, columns=[f"factor_{j + 1}" for j in range(factors_np.shape[1])])
        factors.insert(0, "date", self.train_dates_)
        factors["model"] = self.model_name
        factors["factor_kind"] = "train_factor"
        mean_row: dict[str, Any] = {"date": pd.NaT, "model": self.model_name, "factor_kind": "forecast_mean"}
        for j, value in enumerate(self.factor_mean_, start=1):
            mean_row[f"factor_{j}"] = value
        return pd.concat([factors, pd.DataFrame([mean_row])], ignore_index=True)


class _CAEModel(nn.Module):
    def __init__(self, config: TorchModelConfig, n_dates: int) -> None:
        super().__init__()
        self.beta_net = ConditionalBetaMLP(config)
        self.factor_embeddings = nn.Embedding(n_dates, config.n_factors)
        nn.init.normal_(self.factor_embeddings.weight, mean=0.0, std=0.02)

    def factor_mean(self) -> Tensor:
        return self.factor_embeddings.weight.mean(dim=0)

    def forward(self, x: Tensor, date_idx: Tensor) -> tuple[Tensor, Tensor]:
        beta = self.beta_net(x)
        factors = self.factor_embeddings(date_idx)
        pred = (beta * factors).sum(dim=1)
        return pred, beta

    def predict_with_factor_mean(self, x: Tensor) -> tuple[Tensor, Tensor]:
        beta = self.beta_net(x)
        pred = beta @ self.factor_mean()
        return pred, beta


