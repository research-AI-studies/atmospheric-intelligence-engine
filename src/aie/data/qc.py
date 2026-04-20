"""Quality control for hourly air quality observations.

The QC routine is deliberately conservative: it only flags *physically
implausible* values and interpolates short gaps. It does NOT attempt to
impute long outages, since the downstream models handle missingness via
explicit mask channels in feature engineering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from aie.config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class QCReport:
    """Summary of QC actions applied to a single variable."""

    variable: str
    n_before: int
    n_out_of_range: int
    n_interpolated: int
    n_remaining_missing: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "variable": self.variable,
            "n_before": self.n_before,
            "n_out_of_range": self.n_out_of_range,
            "n_interpolated": self.n_interpolated,
            "n_remaining_missing": self.n_remaining_missing,
        }


def _interpolate_short_gaps(series: pd.Series, max_gap: int) -> tuple[pd.Series, int]:
    """Linearly interpolate gaps with length <= ``max_gap`` hourly steps."""

    missing = series.isna()
    if not missing.any():
        return series, 0

    # Identify contiguous missing runs.
    run_id = (missing != missing.shift()).cumsum()
    runs = series.groupby(run_id).apply(lambda s: (s.isna().all(), len(s)))

    # Mask indicating which missing cells are inside a short gap.
    short_mask = pd.Series(False, index=series.index)
    for rid, (all_missing, length) in runs.items():
        if bool(all_missing) and length <= max_gap:
            short_mask[run_id == rid] = True

    interp = series.interpolate(method="linear", limit=max_gap, limit_area="inside")
    out = series.copy()
    out[short_mask] = interp[short_mask]
    return out, int(short_mask.sum())


def apply_qc(df: pd.DataFrame, cfg: DataConfig) -> tuple[pd.DataFrame, list[QCReport]]:
    """Apply range checks and short-gap interpolation to ``df`` in place (copy)."""

    df = df.copy()
    reports: list[QCReport] = []

    for var, (lo, hi) in cfg.qc_ranges.items():
        if var not in df.columns:
            continue
        series = df[var].astype(float)
        n_before = len(series)
        mask_bad = (series < lo) | (series > hi)
        n_bad = int(mask_bad.sum())
        if n_bad:
            logger.info("QC: %s has %d out-of-range values -> set to NaN", var, n_bad)
            series = series.mask(mask_bad)

        series, n_interp = _interpolate_short_gaps(series, cfg.max_gap_hours)
        n_missing = int(series.isna().sum())

        df[var] = series
        reports.append(
            QCReport(
                variable=var,
                n_before=n_before,
                n_out_of_range=n_bad,
                n_interpolated=n_interp,
                n_remaining_missing=n_missing,
            )
        )

    # Convenience mask column indicating rows where the forecast target is
    # observed (used downstream for loss masking).
    df["is_observed_pm25"] = (~df["pm25"].isna()).astype(np.int8)
    return df, reports
