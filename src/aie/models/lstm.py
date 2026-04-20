"""Vanilla LSTM multi-horizon forecaster (baseline)."""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """Sequence-to-vector LSTM that predicts a vector of length ``len(horizons)``.

    Input tensor shape: ``(batch, input_window, n_features)``.
    Output tensor shape: ``(batch, n_horizons)``.
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.encoder(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))
