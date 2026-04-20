"""Tests for data loader, QC, feature engineering, and splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aie.config import DataConfig, FeatureConfig, SplitConfig
from aie.data import apply_qc, build_features, walk_forward_split


def test_qc_out_of_range_is_masked(synthetic_hourly: pd.DataFrame) -> None:
    df = synthetic_hourly.copy()
    df.loc[5, "pm25"] = 9999.0  # implausible
    cfg = DataConfig()
    out, reports = apply_qc(df, cfg)
    assert np.isnan(out.loc[5, "pm25"]) or out.loc[5, "pm25"] < 9999.0
    pm25_report = next(r for r in reports if r.variable == "pm25")
    assert pm25_report.n_out_of_range >= 1


def test_build_features_adds_expected_columns(synthetic_hourly: pd.DataFrame) -> None:
    cfg = FeatureConfig(target="pm25", lag_hours=[1, 24], rolling_windows=[3])
    feats = build_features(synthetic_hourly, cfg)
    assert "target" in feats.columns
    assert "target_lag_1" in feats.columns
    assert "target_lag_24" in feats.columns
    assert "target_rollmean_3" in feats.columns
    assert "hour_sin" in feats.columns
    assert "wind_u" in feats.columns and "wind_v" in feats.columns


def test_walk_forward_split_disjoint(synthetic_hourly: pd.DataFrame) -> None:
    feats = build_features(synthetic_hourly, FeatureConfig())
    # Extend so we have >1 year
    extra = feats.copy()
    extra["datetime"] = extra["datetime"] + pd.DateOffset(years=1)
    feats = pd.concat([feats, extra], ignore_index=True)
    extra2 = feats.iloc[: len(feats) // 2].copy()
    extra2["datetime"] = extra2["datetime"] + pd.DateOffset(years=2)
    feats = pd.concat([feats, extra2], ignore_index=True)
    cfg = SplitConfig(train_years=[2018], val_years=[2019], test_years=[2020])
    split = walk_forward_split(feats, cfg)
    assert (split.train & split.val).sum() == 0
    assert (split.val & split.test).sum() == 0
    assert split.summary()["n_train"] > 0


def test_walk_forward_split_raises_when_empty(synthetic_hourly: pd.DataFrame) -> None:
    feats = build_features(synthetic_hourly, FeatureConfig())
    with pytest.raises(ValueError):
        walk_forward_split(feats, SplitConfig(train_years=[2099]))
