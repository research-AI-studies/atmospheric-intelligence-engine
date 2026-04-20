"""Pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_hourly() -> pd.DataFrame:
    """Deterministic 3-month hourly DataFrame matching the canonical schema."""

    rng = np.random.default_rng(0)
    ts = pd.date_range("2018-01-01", "2018-03-31 23:00", freq="H")
    n = len(ts)
    hour = ts.hour.to_numpy()
    doy = ts.dayofyear.to_numpy()
    pm25 = 15 + 3 * np.sin(2 * np.pi * hour / 24) + 0.5 * np.sin(2 * np.pi * doy / 366) + rng.normal(0, 2, n)
    pm10 = pm25 * 1.4 + rng.normal(0, 1.5, n)
    df = pd.DataFrame(
        {
            "datetime": ts,
            "station_id": "CA06P",
            "location": "Seberang Jaya",
            "pm10": pm10,
            "pm25": pm25,
            "so2": 0.004 + 0.001 * rng.standard_normal(n),
            "no2": 0.02 + 0.005 * rng.standard_normal(n),
            "o3": np.clip(0.02 + 0.01 * np.sin(2 * np.pi * hour / 24), 0, 0.2),
            "co": 0.9 + 0.1 * rng.standard_normal(n),
            "wind_direction": rng.uniform(0, 360, n),
            "wind_speed": np.clip(1.5 + 0.6 * rng.standard_normal(n), 0, None),
            "relative_humidity": 75 + 10 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 4, n),
            "temperature": 27 + 3 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 1.2, n),
        }
    )
    return df
