"""Atmospheric Intelligence Engine (AIE) architecture.

A temporal convolutional encoder feeds a small Transformer encoder, followed
by a gated residual network head that emits one prediction per requested
horizon. Dropout is kept active at inference time (Monte Carlo dropout) so
that the same module supplies both the point forecast and an ensemble over
stochastic forward passes from which predictive intervals are derived.

The module also exposes an ``adapt`` method that performs a short, bounded
fine-tuning step on the most recent observations, providing the online
recalibration behaviour described in the manuscript.
"""

from __future__ import annotations

import copy

import torch
from torch import nn


class TemporalConvBlock(nn.Module):
    """A causal dilated convolution with gated residual connection."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, T)
        residual = x
        out = torch.relu(self.conv1(x))
        out = out[..., : -self.padding] if self.padding else out
        out = self.dropout(out)
        out = self.conv2(out)
        out = out[..., : -self.padding] if self.padding else out
        out = self.dropout(out)
        out = out + residual
        # Apply LayerNorm over the channel dimension.
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return out


class GatedResidualNetwork(nn.Module):
    """Gated residual network (as in Temporal Fusion Transformers)."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = self.dropout(self.fc2(h))
        gate = torch.sigmoid(self.gate(x))
        return self.norm(x + gate * h)


class AtmosphericIntelligenceEngine(nn.Module):
    """The paper's proposed architecture.

    Parameters
    ----------
    n_features : int
        Number of input features per time step.
    n_horizons : int
        Number of forecast horizons to emit jointly.
    hidden_size : int
        Channel width for the temporal convolution and Transformer.
    num_tcn_blocks : int
        Number of dilated convolution blocks (dilations 1, 2, 4, ...).
    num_transformer_layers : int
        Number of Transformer encoder layers on top of the TCN.
    n_heads : int
        Number of attention heads.
    dropout : float
        Shared dropout rate (used as both training and MC-dropout rate).
    """

    def __init__(
        self,
        n_features: int,
        n_horizons: int,
        hidden_size: int = 128,
        num_tcn_blocks: int = 3,
        num_transformer_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_size)
        self.tcn = nn.ModuleList(
            [
                TemporalConvBlock(hidden_size, kernel_size=3, dilation=2**i, dropout=dropout)
                for i in range(num_tcn_blocks)
            ]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.grn = GatedResidualNetwork(hidden_size, dropout)
        self.head = nn.Linear(hidden_size, n_horizons)
        self._frozen_state: dict[str, torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, F)
        h = self.input_proj(x)  # (B, T, H)
        h_c = h.transpose(1, 2)  # (B, H, T)
        for block in self.tcn:
            h_c = block(h_c)
        h = h_c.transpose(1, 2)  # (B, T, H)
        h = self.transformer(h)
        h = self.grn(h[:, -1, :])
        return self.head(h)

    # ------------------------------------------------------------------
    # MC-dropout predictive sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, n_samples: int = 50) -> torch.Tensor:
        """Return ``(n_samples, batch, n_horizons)`` predictions with dropout active."""

        self.train()  # enable dropout layers
        preds = torch.stack([self.forward(x) for _ in range(n_samples)], dim=0)
        self.eval()
        return preds

    # ------------------------------------------------------------------
    # Online recalibration
    # ------------------------------------------------------------------
    def snapshot(self) -> None:
        """Store a deep copy of the current weights to enable safe rollback."""

        self._frozen_state = copy.deepcopy(self.state_dict())

    def restore(self) -> None:
        """Restore the weights saved by :meth:`snapshot`."""

        if self._frozen_state is None:
            raise RuntimeError("No snapshot has been taken.")
        self.load_state_dict(self._frozen_state)

    def adapt(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor | None = None,
        n_steps: int = 5,
        lr: float = 1.0e-4,
    ) -> float:
        """Run a short, bounded gradient update on recent ``(x, y)`` pairs.

        Performs a small number of optimizer steps on the most recent batch
        so that the model can track slow distributional drift such as
        seasonal shifts in baseline PM levels. The caller should invoke
        :meth:`snapshot` before calling ``adapt`` and :meth:`restore` if
        the update fails a validation check.

        Returns
        -------
        float
            Final training-loss value after adaptation.
        """

        self.train()
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction="none")
        last_loss = float("nan")
        for _ in range(n_steps):
            optim.zero_grad()
            pred = self.forward(x)
            losses = loss_fn(pred, y)
            if mask is not None:
                losses = losses * mask
                denom = mask.sum().clamp(min=1.0)
                loss = losses.sum() / denom
            else:
                loss = losses.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            optim.step()
            last_loss = float(loss.item())
        return last_loss
