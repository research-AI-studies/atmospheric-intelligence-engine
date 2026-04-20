"""Strict time-ordered (walk-forward) splits.

No future information leaks backwards: the training set contains strictly
earlier timestamps than the validation set, which in turn is strictly earlier
than the test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aie.config import SplitConfig


@dataclass
class Split:
    """Boolean masks indexed by the feature DataFrame rows."""

    train: pd.Series
    val: pd.Series
    test: pd.Series

    def summary(self) -> dict[str, int]:
        return {
            "n_train": int(self.train.sum()),
            "n_val": int(self.val.sum()),
            "n_test": int(self.test.sum()),
        }


def walk_forward_split(feature_df: pd.DataFrame, cfg: SplitConfig) -> Split:
    """Build boolean masks based on the `datetime` column and configured years."""

    years = feature_df["datetime"].dt.year
    train = years.isin(cfg.train_years)
    val = years.isin(cfg.val_years)
    test = years.isin(cfg.test_years)

    if not train.any():
        raise ValueError(f"No rows in train years {cfg.train_years}")
    if not val.any():
        raise ValueError(f"No rows in val years {cfg.val_years}")
    if not test.any():
        raise ValueError(f"No rows in test years {cfg.test_years}")

    return Split(train=train, val=val, test=test)
