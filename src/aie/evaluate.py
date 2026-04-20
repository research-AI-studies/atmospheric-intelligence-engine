"""Forecast-skill metrics and walk-forward evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (y_true - y_pred) ** 2
    if mask is not None:
        diff = diff[mask.astype(bool)]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(np.sqrt(diff.mean()))


def mae(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(y_true - y_pred)
    if mask is not None:
        diff = diff[mask.astype(bool)]
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    return float(diff.mean())


def r2(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        m = mask.astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size < 2:
        return float("nan")
    ss_res = float(((y_true - y_pred) ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def index_of_agreement(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Willmott's index of agreement (d)."""

    if mask is not None:
        m = mask.astype(bool)
        y_true = y_true[m]
        y_pred = y_pred[m]
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[finite]
    y_pred = y_pred[finite]
    if y_true.size < 2:
        return float("nan")
    ybar = y_true.mean()
    num = float(((y_pred - y_true) ** 2).sum())
    den = float(((np.abs(y_pred - ybar) + np.abs(y_true - ybar)) ** 2).sum())
    if den == 0:
        return float("nan")
    return 1.0 - num / den


@dataclass
class HorizonMetrics:
    horizon: int
    rmse: float
    mae: float
    r2: float
    ioa: float
    n: int


def per_horizon_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int], mask: np.ndarray | None = None
) -> pd.DataFrame:
    """Compute RMSE / MAE / R^2 / IOA per horizon."""

    rows: list[HorizonMetrics] = []
    for j, h in enumerate(horizons):
        m = mask[:, j] if mask is not None else None
        rows.append(
            HorizonMetrics(
                horizon=h,
                rmse=rmse(y_true[:, j], y_pred[:, j], m),
                mae=mae(y_true[:, j], y_pred[:, j], m),
                r2=r2(y_true[:, j], y_pred[:, j], m),
                ioa=index_of_agreement(y_true[:, j], y_pred[:, j], m),
                n=int(np.isfinite(y_true[:, j]).sum() if m is None else m.astype(bool).sum()),
            )
        )
    return pd.DataFrame([r.__dict__ for r in rows])
