"""Tests for evaluation metrics and uncertainty diagnostics."""

from __future__ import annotations

import numpy as np

from aie.evaluate import index_of_agreement, mae, per_horizon_metrics, r2, rmse
from aie.uncertainty import reliability_diagram


def test_rmse_mae_r2_trivial() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    yhat = y.copy()
    assert rmse(y, yhat) == 0.0
    assert mae(y, yhat) == 0.0
    assert r2(y, yhat) == 1.0
    assert index_of_agreement(y, yhat) == 1.0


def test_per_horizon_metrics_shape() -> None:
    n, H = 100, 3
    rng = np.random.default_rng(0)
    y = rng.normal(size=(n, H))
    yhat = y + rng.normal(scale=0.3, size=(n, H))
    mask = np.ones_like(y)
    df = per_horizon_metrics(y, yhat, horizons=[1, 6, 24], mask=mask)
    assert list(df.columns) == ["horizon", "rmse", "mae", "r2", "ioa", "n"]
    assert len(df) == H


def test_reliability_diagram_monotone_for_gaussian() -> None:
    rng = np.random.default_rng(0)
    n = 5000
    mu = rng.normal(size=n)
    sigma = np.abs(rng.normal(size=n)) + 0.5
    y = rng.normal(loc=mu, scale=sigma)
    levels = np.linspace(0.1, 0.9, 9)
    cov = reliability_diagram(y, mu, sigma, levels)
    assert np.all(np.diff(cov) >= -0.05)
