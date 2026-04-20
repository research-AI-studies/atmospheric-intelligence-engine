"""Publication-ready figure helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PUBLICATION_RCPARAMS = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def apply_style() -> None:
    plt.rcParams.update(PUBLICATION_RCPARAMS)
    sns.set_palette("colorblind")


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".png"))
    fig.savefig(p.with_suffix(".pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------


def plot_missingness_heatmap(df: pd.DataFrame, out_path: str | Path) -> None:
    cols = [c for c in df.columns if c not in {"datetime", "station_id", "location", "sheet_year"}]
    grid = (
        df.set_index("datetime")[cols]
        .resample("1D")
        .apply(lambda s: s.isna().mean())
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(grid.T, cmap="magma", vmin=0, vmax=1, cbar_kws={"label": "fraction missing"}, ax=ax)
    ax.set_title("Data availability (daily fraction missing) - Seberang Jaya 2018-2021")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, out_path)


def plot_diurnal_cycle(df: pd.DataFrame, variable: str, out_path: str | Path) -> None:
    d = df[["datetime", variable]].dropna().copy()
    d["hour"] = d["datetime"].dt.hour
    d["month"] = d["datetime"].dt.month
    agg = d.groupby(["month", "hour"])[variable].median().reset_index()
    pivot = agg.pivot(index="month", columns="hour", values=variable)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot, cmap="viridis", cbar_kws={"label": variable}, ax=ax)
    ax.set_title(f"Median {variable} by month and hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Month")
    save_figure(fig, out_path)


def plot_correlation_matrix(df: pd.DataFrame, out_path: str | Path) -> None:
    cols = [
        c
        for c in ("pm10", "pm25", "so2", "no2", "o3", "co", "wind_speed", "temperature", "relative_humidity")
        if c in df.columns
    ]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation among pollutants and meteorology")
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Forecast skill
# ---------------------------------------------------------------------------


def plot_skill_vs_horizon(metric_tables: dict[str, pd.DataFrame], metric: str, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, df in metric_tables.items():
        ax.plot(df["horizon"], df[metric], marker="o", label=name)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Forecast {metric.upper()} vs. horizon - Seberang Jaya 2021 test year")
    ax.set_xscale("log")
    ax.legend()
    save_figure(fig, out_path)


def plot_predicted_vs_observed(
    y_true: np.ndarray, y_pred: np.ndarray, horizon: int, out_path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    ax.scatter(y_true[finite], y_pred[finite], s=5, alpha=0.3)
    lim = float(max(y_true[finite].max(), y_pred[finite].max()))
    ax.plot([0, lim], [0, lim], "k--", linewidth=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Predicted vs. observed (h = {horizon} h)")
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def plot_reliability_diagram(
    coverage: np.ndarray, levels: np.ndarray, out_path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(levels, coverage, marker="o", label="AIE (MC-dropout)")
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Reliability diagram - AIE predictive intervals")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    save_figure(fig, out_path)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def plot_scenarios(results: list, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in results:
        ax.plot(r.timestamps, r.mean, label=r.label, linewidth=1.2)
        ax.fill_between(r.timestamps, r.p10, r.p90, alpha=0.15)
    ax.set_xlabel("Time (hourly)")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_title("Scenario sensitivity to assumed drivers - annual roll-out")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    save_figure(fig, out_path)
