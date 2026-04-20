"""Baseline forecasters: persistence and XGBoost direct multi-horizon."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


@dataclass
class PersistenceModel:
    """Trivial baseline: prediction at horizon ``h`` equals the last observed value."""

    horizons: list[int]

    def fit(self, *_, **__) -> "PersistenceModel":  # noqa: D401 - trivial
        return self

    def predict(self, targets: np.ndarray) -> np.ndarray:
        """Return an ``(n, len(horizons))`` array of persistence forecasts.

        ``targets`` is the observed target series aligned with the feature
        DataFrame; for every step we simply repeat the current value across
        all horizons.
        """

        preds = np.repeat(targets.reshape(-1, 1), len(self.horizons), axis=1)
        return preds


# ---------------------------------------------------------------------------
# XGBoost (one model per horizon)
# ---------------------------------------------------------------------------


class XGBoostForecaster:
    """Direct multi-horizon XGBoost regressor.

    One independent booster is trained per horizon; at horizon ``h`` the
    training target is ``y[t + h]`` aligned with features ``X[t]``.
    """

    def __init__(
        self,
        horizons: list[int],
        n_estimators: int = 400,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
    ) -> None:
        self.horizons = horizons
        self.boosters: dict[int, xgb.XGBRegressor] = {}
        self.feature_names: list[str] | None = None
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            tree_method="hist",
            objective="reg:squarederror",
        )

    @staticmethod
    def _shift_target(y: np.ndarray, horizon: int) -> np.ndarray:
        shifted = np.roll(y, -horizon)
        shifted[-horizon:] = np.nan
        return shifted

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame | None = None,
        y_val: np.ndarray | None = None,
    ) -> "XGBoostForecaster":
        self.feature_names = list(X_train.columns)
        for h in self.horizons:
            y_h = self._shift_target(y_train, h)
            mask = ~np.isnan(y_h) & ~np.isnan(X_train.values).any(axis=1)
            booster = xgb.XGBRegressor(**self._params)
            eval_set = None
            if X_val is not None and y_val is not None:
                y_v = self._shift_target(y_val, h)
                vmask = ~np.isnan(y_v) & ~np.isnan(X_val.values).any(axis=1)
                if vmask.any():
                    eval_set = [(X_val.values[vmask], y_v[vmask])]
            booster.fit(
                X_train.values[mask],
                y_h[mask],
                eval_set=eval_set,
                verbose=False,
            )
            self.boosters[h] = booster
            logger.info("XGBoost horizon %d: trained on %d rows", h, mask.sum())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        out = np.full((len(X), len(self.horizons)), np.nan, dtype=float)
        Xv = X.values
        row_valid = ~np.isnan(Xv).any(axis=1)
        for j, h in enumerate(self.horizons):
            booster = self.boosters[h]
            out[row_valid, j] = booster.predict(Xv[row_valid])
        return out
