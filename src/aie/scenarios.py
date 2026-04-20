"""Scenario analysis under perturbed meteorological drivers.

This module produces sensitivity trajectories by rolling a trained model
forward on bootstrapped historical meteorology with user-specified
perturbations of temperature and emission proxies. Reported bands summarise
the spread across MC-dropout samples and scenario members, characterising
sensitivity to the assumed driver perturbations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    label: str
    timestamps: pd.DatetimeIndex
    mean: np.ndarray
    p10: np.ndarray
    p90: np.ndarray


def _bootstrap_meteorology(
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    n_steps: int,
    seed: int = 0,
) -> np.ndarray:
    """Sample an ``n_steps`` hourly sequence by block-bootstrapping history.

    Blocks of 24 hours are drawn at random (with replacement) from the full
    record to build a synthetic meteorology sequence with realistic diurnal
    structure.
    """

    rng = np.random.default_rng(seed)
    vals = feature_df[feature_cols].to_numpy(dtype=np.float32)
    vals = np.nan_to_num(vals, nan=0.0)
    n = len(vals)
    block = 24
    starts = rng.integers(0, n - block, size=(n_steps // block) + 1)
    chunks = [vals[s : s + block] for s in starts]
    seq = np.concatenate(chunks, axis=0)[:n_steps]
    return seq


def _apply_perturbation(
    seq: np.ndarray,
    feature_cols: list[str],
    temperature_delta_c: float,
    emission_scale: float,
) -> np.ndarray:
    """Apply temperature and emission-proxy perturbations in-place on a copy."""

    seq = seq.copy()
    idx = {c: i for i, c in enumerate(feature_cols)}
    if "temperature" in idx:
        seq[:, idx["temperature"]] += temperature_delta_c
    for col in ("pm10", "pm25", "no2", "so2", "co"):
        if col in idx:
            seq[:, idx[col]] *= emission_scale
    return seq


def run_scenarios(
    model: torch.nn.Module,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    input_window: int,
    temperature_deltas: list[float],
    emission_scales: list[float],
    horizon_hours: int,
    n_members: int,
    device: torch.device,
    start_timestamp: pd.Timestamp | None = None,
) -> list[ScenarioResult]:
    """Produce scenario trajectories for every (temperature, emission) combination."""

    if start_timestamp is None:
        start_timestamp = feature_df["datetime"].max() + pd.Timedelta(hours=1)

    ts_index = pd.date_range(start_timestamp, periods=horizon_hours, freq="h")
    model.eval()
    # Keep dropout stochasticity for ensemble spread.
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()

    results: list[ScenarioResult] = []
    h_max = max(horizons)
    for dT in temperature_deltas:
        for es in emission_scales:
            label = f"dT={dT:+.1f}C,E={es:.2f}"
            member_trajectories: list[np.ndarray] = []
            for member in range(n_members):
                seq = _bootstrap_meteorology(
                    feature_df, feature_cols, horizon_hours + input_window, seed=member
                )
                seq = _apply_perturbation(seq, feature_cols, dT, es)
                x = torch.from_numpy(seq[-(input_window + horizon_hours) : -horizon_hours]).to(
                    device
                )
                x = x.unsqueeze(0)  # (1, T, F)
                with torch.no_grad():
                    pred = model(x).cpu().numpy().squeeze(0)  # (len(horizons),)
                # Piecewise-constant broadcast across horizons: conservative
                # temporal structure that matches the direct multi-horizon head.
                traj = np.full(horizon_hours, np.nan, dtype=float)
                for h_i, h in enumerate(horizons):
                    start = max(0, h - 1)
                    end = h_max if h == h_max else horizons[h_i + 1] - 1
                    traj[start:end] = pred[h_i]
                # Forward-fill to fill any remaining NaNs.
                s = pd.Series(traj).ffill().bfill().to_numpy()
                member_trajectories.append(s)
            arr = np.stack(member_trajectories, axis=0)  # (M, T)
            results.append(
                ScenarioResult(
                    label=label,
                    timestamps=ts_index,
                    mean=arr.mean(axis=0),
                    p10=np.percentile(arr, 10, axis=0),
                    p90=np.percentile(arr, 90, axis=0),
                )
            )
            logger.info("scenario %s: mean=%.2f", label, arr.mean())
    model.eval()
    return results
