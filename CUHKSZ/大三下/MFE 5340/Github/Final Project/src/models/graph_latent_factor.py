"""Graph-enhanced conditional latent factor model for Stage 5."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv, GCNConv


@dataclass(frozen=True)
class GraphLatentFactorConfig:
    """Model dimensions and graph encoder choices."""

    input_dim: int
    hidden_dim: int = 64
    n_factors: int = 3
    dropout: float = 0.1
    model_type: str = "gcn"
    gat_heads: int = 2


class GraphBetaEncoder(nn.Module):
    """Map month-t node features and graph context to conditional exposures."""

    def __init__(self, config: GraphLatentFactorConfig) -> None:
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        if config.model_type == "gcn":
            self.conv1 = GCNConv(config.input_dim, config.hidden_dim)
            self.conv2 = GCNConv(config.hidden_dim, config.n_factors)
        elif config.model_type == "gat":
            self.conv1 = GATConv(config.input_dim, config.hidden_dim, heads=config.gat_heads, dropout=config.dropout)
            self.conv2 = GATConv(config.hidden_dim * config.gat_heads, config.n_factors, heads=1, concat=False, dropout=config.dropout)
        else:
            raise ValueError("model_type must be 'gcn' or 'gat'")
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Return conditional beta and optional final-layer GAT attention."""

        if self.model_type == "gcn":
            hidden = self.conv1(x, edge_index, edge_weight=edge_weight)
            hidden = self.dropout(self.activation(hidden))
            beta = self.conv2(hidden, edge_index, edge_weight=edge_weight)
            return beta, None

        hidden = self.conv1(x, edge_index)
        hidden = self.dropout(self.activation(hidden))
        if return_attention:
            beta, attention = self.conv2(hidden, edge_index, return_attention_weights=True)
            return beta, attention
        beta = self.conv2(hidden, edge_index)
        return beta, None


class GraphConditionalLatentFactorModel(nn.Module):
    """Graph conditional beta model with train-window latent factor embeddings."""

    model_name = "graph_conditional_latent_factor"

    def __init__(self, config: GraphLatentFactorConfig, n_train_dates: int) -> None:
        super().__init__()
        self.config = config
        self.beta_encoder = GraphBetaEncoder(config)
        self.factor_embeddings = nn.Embedding(n_train_dates, config.n_factors)
        nn.init.normal_(self.factor_embeddings.weight, mean=0.0, std=0.02)

    def factor_mean(self) -> Tensor:
        """Historical mean latent factor premium used for OOS prediction."""

        return self.factor_embeddings.weight.mean(dim=0)

    def beta(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Estimate graph-informed conditional exposures."""

        return self.beta_encoder(x, edge_index, edge_weight=edge_weight, return_attention=return_attention)

    def reconstruct_with_train_factor(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None,
        date_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Reconstruct train-month returns using date-specific latent factors."""

        beta, _attention = self.beta(x, edge_index, edge_weight=edge_weight, return_attention=False)
        factor = self.factor_embeddings(date_idx).squeeze(0)
        pred = beta @ factor
        return pred, beta

    def predict_with_factor_mean(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        """Predict OOS returns with graph-informed beta and mean factor premium."""

        beta, attention = self.beta(x, edge_index, edge_weight=edge_weight, return_attention=return_attention)
        pred = beta @ self.factor_mean()
        return pred, beta, attention
