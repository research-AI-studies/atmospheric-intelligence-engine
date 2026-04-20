"""PyTorch Dataset for hourly sliding-window forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Turn a flat feature DataFrame into ``(x, y, mask)`` windows.

    Parameters
    ----------
    features : pandas.DataFrame
        Rows are consecutive hourly observations (possibly with NaNs in the
        target). A ``target`` column must be present.
    feature_cols : list[str]
        Names of the columns to use as model input.
    horizons : list[int]
        Future offsets (hours) to predict jointly.
    input_window : int
        Number of past hours used as input.
    """

    def __init__(
        self,
        features: pd.DataFrame,
        feature_cols: list[str],
        horizons: list[int],
        input_window: int,
    ) -> None:
        if "target" not in features.columns:
            raise KeyError("features DataFrame must contain a 'target' column.")
        self.horizons = sorted(horizons)
        self.input_window = input_window
        self._max_h = max(self.horizons)

        # Replace NaNs in inputs with zeros (missingness mask is included as a
        # feature, so the model can learn to ignore imputed zeros).
        X_full = features[feature_cols].to_numpy(dtype=np.float32)
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        y_full = features["target"].to_numpy(dtype=np.float32)
        y_mask = (~np.isnan(y_full)).astype(np.float32)

        usable_start = self.input_window
        usable_end = len(features) - self._max_h
        if usable_end <= usable_start:
            raise ValueError(
                f"Not enough rows to build windows: need > {self.input_window + self._max_h}, "
                f"have {len(features)}."
            )

        self.X = X_full
        self.y = np.nan_to_num(y_full, nan=0.0)
        self.mask = y_mask
        self._indices = np.arange(usable_start, usable_end, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = int(self._indices[idx])
        x = self.X[t - self.input_window : t]
        y = np.array([self.y[t + h - 1] for h in self.horizons], dtype=np.float32)
        m = np.array([self.mask[t + h - 1] for h in self.horizons], dtype=np.float32)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(m),
        )

    @property
    def anchor_indices(self) -> np.ndarray:
        """Row indices in the original feature DataFrame that each window anchors on."""

        return self._indices.copy()
