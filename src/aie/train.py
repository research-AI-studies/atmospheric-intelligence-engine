"""Training loops for LSTM and AIE models with early stopping."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aie.config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    diff = diff * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ModelConfig,
    device: torch.device,
) -> tuple[nn.Module, TrainingHistory]:
    """Generic supervised training loop with early stopping.

    The target is assumed to be pre-standardised (see :class:`aie.training.Pipeline`).
    Missing target entries are masked out of the loss.
    """

    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    history = TrainingHistory(train_loss=[], val_loss=[], best_epoch=-1, best_val_loss=float("inf"))
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_losses: list[float] = []
        for x, y, m in train_loader:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            optim.zero_grad()
            pred = model(x)
            loss = _masked_mse(pred, y, m)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for x, y, m in val_loader:
                x = x.to(device)
                y = y.to(device)
                m = m.to(device)
                pred = model(x)
                val_losses.append(float(_masked_mse(pred, y, m).item()))

        tl = float(np.mean(train_losses)) if train_losses else float("nan")
        vl = float(np.mean(val_losses)) if val_losses else float("nan")
        history.train_loss.append(tl)
        history.val_loss.append(vl)
        logger.info("epoch %02d  train=%.4f  val=%.4f", epoch, tl, vl)

        if vl < history.best_val_loss - 1.0e-6:
            history.best_val_loss = vl
            history.best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
