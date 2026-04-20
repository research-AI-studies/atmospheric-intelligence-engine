"""MC-dropout uncertainty quantification and calibration diagnostics."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def mc_dropout_forecast(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run MC-dropout forecasting over ``loader``.

    Returns
    -------
    y_true : ndarray, shape (N, H)
    y_mean : ndarray, shape (N, H)
    y_std  : ndarray, shape (N, H)
    mask   : ndarray, shape (N, H)
    """

    model.eval()
    # Force dropout layers to remain active.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    all_true: list[np.ndarray] = []
    all_mean: list[np.ndarray] = []
    all_std: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []

    for x, y, mask in loader:
        x = x.to(device)
        preds = torch.stack([model(x) for _ in range(n_samples)], dim=0)  # (S, B, H)
        mu = preds.mean(dim=0).cpu().numpy()
        sd = preds.std(dim=0).cpu().numpy()
        all_true.append(y.numpy())
        all_mean.append(mu)
        all_std.append(sd)
        all_mask.append(mask.numpy())

    model.eval()
    return (
        np.concatenate(all_true, axis=0),
        np.concatenate(all_mean, axis=0),
        np.concatenate(all_std, axis=0),
        np.concatenate(all_mask, axis=0),
    )


def reliability_diagram(
    y_true: np.ndarray, y_mean: np.ndarray, y_std: np.ndarray, levels: np.ndarray | None = None
) -> np.ndarray:
    """Return empirical coverage at nominal confidence levels.

    Assumes Gaussian predictive distribution ``N(y_mean, y_std**2)``. For each
    nominal level ``p``, counts the fraction of true values that fall inside
    the symmetric ``p`` interval.
    """

    if levels is None:
        levels = np.linspace(0.1, 0.95, 18)
    from scipy.stats import norm

    coverage = np.zeros_like(levels, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_mean) & np.isfinite(y_std) & (y_std > 0)
    y_true = y_true[finite]
    y_mean = y_mean[finite]
    y_std = y_std[finite]
    if y_true.size == 0:
        return np.full_like(levels, np.nan, dtype=float)

    z = (y_true - y_mean) / y_std
    for i, p in enumerate(levels):
        half = norm.ppf(0.5 + p / 2.0)
        coverage[i] = float((np.abs(z) <= half).mean())
    return coverage


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Mean CRPS assuming Gaussian predictive distributions."""

    try:
        import properscoring as ps  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("properscoring is required for CRPS") from exc
    finite = np.isfinite(y_true) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0)
    if not finite.any():
        return float("nan")
    return float(ps.crps_gaussian(y_true[finite], mu=mu[finite], sig=sigma[finite]).mean())
