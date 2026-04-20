"""Feature engineering for hourly air quality forecasting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from aie.config import FeatureConfig

logger = logging.getLogger(__name__)


POLLUTANTS: list[str] = ["pm10", "pm25", "so2", "no2", "o3", "co"]
METEOROLOGY: list[str] = [
    "wind_speed",
    "wind_direction",
    "relative_humidity",
    "temperature",
]


def _wind_uv(wind_speed: pd.Series, wind_direction: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Decompose wind speed + direction (meteorological convention) into u, v."""

    rad = np.deg2rad(wind_direction.astype(float))
    u = -wind_speed.astype(float) * np.sin(rad)
    v = -wind_speed.astype(float) * np.cos(rad)
    return u.rename("wind_u"), v.rename("wind_v")


def _calendar_features(ts: pd.Series) -> pd.DataFrame:
    hour = ts.dt.hour
    doy = ts.dt.dayofyear
    dow = ts.dt.dayofweek
    month = ts.dt.month

    def _cyc(x: pd.Series, period: int, name: str) -> pd.DataFrame:
        rad = 2 * np.pi * x / period
        return pd.DataFrame({f"{name}_sin": np.sin(rad), f"{name}_cos": np.cos(rad)})

    feats = pd.concat(
        [
            _cyc(hour, 24, "hour"),
            _cyc(doy, 366, "doy"),
            _cyc(dow, 7, "dow"),
            _cyc(month, 12, "month"),
        ],
        axis=1,
    )
    feats["is_weekend"] = (dow >= 5).astype(np.int8)
    return feats


def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return a feature table aligned to ``df['datetime']``.

    The target column is renamed ``target`` and kept as-is; lag, rolling,
    wind-component, and calendar features are appended. Rows where the target
    is NaN are retained (the training loop masks them explicitly).
    """

    if cfg.target not in df.columns:
        raise KeyError(f"Target '{cfg.target}' not found in DataFrame.")

    out = pd.DataFrame({"datetime": df["datetime"].values})
    out["target"] = df[cfg.target].astype(float).values

    # Raw pollutant + meteorology columns
    for col in POLLUTANTS + METEOROLOGY:
        if col in df.columns:
            out[col] = df[col].astype(float).values

    # Wind components
    if cfg.include_wind_components and {"wind_speed", "wind_direction"}.issubset(df.columns):
        u, v = _wind_uv(df["wind_speed"], df["wind_direction"])
        out["wind_u"] = u.values
        out["wind_v"] = v.values

    # Lag features of the target
    tgt = out["target"]
    for lag in cfg.lag_hours:
        out[f"target_lag_{lag}"] = tgt.shift(lag).values

    # Rolling statistics of the target
    for w in cfg.rolling_windows:
        out[f"target_rollmean_{w}"] = tgt.shift(1).rolling(w, min_periods=1).mean().values
        out[f"target_rollstd_{w}"] = tgt.shift(1).rolling(w, min_periods=2).std().values

    # Calendar features
    if cfg.include_calendar:
        cal = _calendar_features(pd.Series(out["datetime"]))
        out = pd.concat([out.reset_index(drop=True), cal.reset_index(drop=True)], axis=1)

    # Missingness indicator on the target
    out["target_is_nan"] = out["target"].isna().astype(np.int8)

    logger.info("Built feature table with %d columns over %d rows.", out.shape[1], len(out))
    return out
