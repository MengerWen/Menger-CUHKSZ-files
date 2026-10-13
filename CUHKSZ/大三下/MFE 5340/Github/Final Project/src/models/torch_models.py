"""Torch model skeletons for later conditional factor and graph experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv


@dataclass(frozen=True)
class TorchModelConfig:
    """Shared neural model dimensions."""

    input_dim: int
    hidden_dim: int = 128
    output_dim: int = 1
    n_factors: int = 5
    dropout: float = 0.1


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    """Create a compact MLP block with torch layers."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class MLPReturnPredictor(nn.Module):
    """Direct next-month return predictor benchmark."""

    def __init__(self, config: TorchModelConfig) -> None:
        super().__init__()
        self.config = config
        self.net = make_mlp(config.input_dim, config.hidden_dim, config.output_dim, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


class ConditionalBetaMLP(nn.Module):
    """Non-graph conditional beta network for latent factor pricing."""

    def __init__(self, config: TorchModelConfig) -> None:
        super().__init__()
        self.config = config
        self.beta_net = make_mlp(config.input_dim, config.hidden_dim, config.n_factors, config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        beta = self.beta_net(x)
        return rearrange(beta, "asset factor -> asset factor")


class GraphConditionalEncoder(nn.Module):
    """GCN/GAT encoder for graph-enhanced conditional exposure learning."""

    def __init__(self, config: TorchModelConfig, model_type: str = "gcn", heads: int = 2) -> None:
        super().__init__()
        self.config = config
        self.model_type = model_type
        if model_type == "gcn":
            self.conv1 = GCNConv(config.input_dim, config.hidden_dim)
            self.conv2 = GCNConv(config.hidden_dim, config.n_factors)
        elif model_type == "gat":
            self.conv1 = GATConv(config.input_dim, config.hidden_dim, heads=heads, dropout=config.dropout)
            self.conv2 = GATConv(config.hidden_dim * heads, config.n_factors, heads=1, concat=False, dropout=config.dropout)
        else:
            raise ValueError("model_type must be 'gcn' or 'gat'")
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor | None = None) -> Tensor:
        if self.model_type == "gcn":
            hidden = self.conv1(x, edge_index, edge_weight=edge_weight)
            hidden = self.dropout(self.activation(hidden))
            return self.conv2(hidden, edge_index, edge_weight=edge_weight)
        hidden = self.conv1(x, edge_index)
        hidden = self.dropout(self.activation(hidden))
        return self.conv2(hidden, edge_index)
