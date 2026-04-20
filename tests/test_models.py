"""Smoke tests for models and training loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aie.config import FeatureConfig, ModelConfig
from aie.data import build_features
from aie.dataset import SlidingWindowDataset
from aie.models import (
    AtmosphericIntelligenceEngine,
    LSTMForecaster,
    PersistenceModel,
    XGBoostForecaster,
)
from aie.train import train_model


def test_persistence_shape(synthetic_hourly: pd.DataFrame) -> None:
    model = PersistenceModel(horizons=[1, 6, 24])
    targets = synthetic_hourly["pm25"].to_numpy(dtype=float)
    preds = model.predict(targets)
    assert preds.shape == (len(targets), 3)


def test_xgboost_fit_predict(synthetic_hourly: pd.DataFrame) -> None:
    feats = build_features(synthetic_hourly, FeatureConfig(lag_hours=[1, 3], rolling_windows=[3]))
    feature_cols = [c for c in feats.columns if c not in {"datetime", "target"}]
    X = feats[feature_cols].fillna(0.0)
    y = feats["target"].ffill().to_numpy(dtype=float)
    model = XGBoostForecaster(horizons=[1, 6], n_estimators=20, max_depth=3, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X), 2)
    assert np.isfinite(preds).all()


def test_lstm_forward_pass(synthetic_hourly: pd.DataFrame) -> None:
    model = LSTMForecaster(n_features=8, n_horizons=3, hidden_size=16, num_layers=1)
    x = torch.randn(4, 24, 8)
    out = model(x)
    assert out.shape == (4, 3)


def test_aie_forward_and_mc(synthetic_hourly: pd.DataFrame) -> None:
    model = AtmosphericIntelligenceEngine(
        n_features=8,
        n_horizons=2,
        hidden_size=16,
        num_tcn_blocks=2,
        num_transformer_layers=1,
        n_heads=2,
    )
    x = torch.randn(4, 24, 8)
    out = model(x)
    assert out.shape == (4, 2)
    mc = model.mc_predict(x, n_samples=5)
    assert mc.shape == (5, 4, 2)


def test_training_loop_runs(synthetic_hourly: pd.DataFrame) -> None:
    feats = build_features(synthetic_hourly, FeatureConfig(lag_hours=[1, 3], rolling_windows=[3]))
    feature_cols = [c for c in feats.columns if c not in {"datetime", "target"}]
    # Normalise to keep training stable.
    normed = feats.copy()
    for c in [*feature_cols, "target"]:
        s = normed[c].astype(float)
        mu, sd = float(np.nanmean(s)), float(np.nanstd(s)) or 1.0
        normed[c] = (s - mu) / sd

    ds = SlidingWindowDataset(normed, feature_cols, horizons=[1, 6], input_window=24)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = LSTMForecaster(n_features=len(feature_cols), n_horizons=2, hidden_size=16, num_layers=1)
    cfg = ModelConfig(
        name="lstm", horizons=[1, 6], input_window=24, epochs=2, patience=2, batch_size=32, lr=0.01
    )
    model, hist = train_model(model, loader, val_loader, cfg, torch.device("cpu"))
    assert len(hist.train_loss) >= 1
