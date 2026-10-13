"""Benchmark registry with pandas metadata and torch backend hints."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class BenchmarkSpec:
    """High-level benchmark definition for experiment configs."""

    name: str
    family: str
    uses_graph: bool
    asset_pricing_role: str
    backend: str
    module_hint: str


BENCHMARKS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec("ipca_style", "linear latent factor", False, "conditional beta benchmark", "sklearn/numpy", "IPCAStyleBenchmark"),
    BenchmarkSpec("conditional_autoencoder_style", "nonlinear latent factor", False, "nonlinear conditional beta benchmark", "torch", "CAEStyleBenchmark"),
    BenchmarkSpec("mlp_predictor", "direct prediction", False, "non-graph nonlinear prediction benchmark", "torch", "MLPReturnPredictor"),
    BenchmarkSpec("graph_conditional_latent_factor", "graph latent factor", True, "graph-enhanced conditional beta model", "torch_geometric", "GraphConditionalLatentFactorModel"),
    BenchmarkSpec("gcn_latent_factor", "graph latent factor", True, "graph-enhanced conditional beta variant", "torch_geometric", "GraphConditionalLatentFactorModel(model_type='gcn')"),
    BenchmarkSpec("gat_latent_factor", "graph latent factor", True, "attention-based graph conditional beta variant", "torch_geometric", "GraphConditionalLatentFactorModel(model_type='gat')"),
)


def benchmark_frame() -> pd.DataFrame:
    """Return benchmark metadata as a pandas DataFrame."""

    return pd.DataFrame([spec.__dict__ for spec in BENCHMARKS])


def benchmark_registry() -> dict[str, BenchmarkSpec]:
    """Return benchmarks keyed by name for config lookup."""

    return {spec.name: spec for spec in BENCHMARKS}


class BenchmarkModel(BaseEstimator):
    """Minimal scikit-learn-style estimator interface for non-torch baselines."""

    def fit(self, *_args: object, **_kwargs: object) -> "BenchmarkModel":
        raise NotImplementedError("Model fitting is reserved for the modeling stage.")

    def predict(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError("Prediction is reserved for the modeling stage.")
