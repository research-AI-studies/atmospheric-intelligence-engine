"""Forecasting models used in the paper."""

from __future__ import annotations

from aie.models.aie import AtmosphericIntelligenceEngine
from aie.models.baselines import PersistenceModel, XGBoostForecaster
from aie.models.lstm import LSTMForecaster

__all__ = [
    "AtmosphericIntelligenceEngine",
    "LSTMForecaster",
    "PersistenceModel",
    "XGBoostForecaster",
]
