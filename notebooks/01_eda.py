# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3 (aie)
#     language: python
#     name: aie
# ---

# %% [markdown]
# # 01. Exploratory Data Analysis
#
# Inspect the Seberang Jaya 2018-2021 hourly record: data availability,
# diurnal / seasonal structure, correlations, and the 2019 Southeast Asian
# haze episode.

# %%
from pathlib import Path

import pandas as pd

from aie.data import apply_qc, load_raw_excel
from aie.config import DataConfig
from aie.plotting import apply_style, plot_missingness_heatmap, plot_diurnal_cycle, plot_correlation_matrix

apply_style()

raw = Path("../data/raw/Seberang Jaya, Pulau Pinang_AIR QUALITY 2018-2021.xlsx")
if not raw.exists():
    raw = Path("../data/sample/synthetic_seberang_jaya.xlsx")

df = load_raw_excel(raw)
df, reports = apply_qc(df, DataConfig())

# %%
print(df.describe().T[["count", "mean", "std", "min", "50%", "max"]].round(3))

# %%
plot_missingness_heatmap(df, "../figures/notebooks/01_missingness")

# %%
for var in ("pm25", "pm10", "o3", "no2"):
    if var in df.columns:
        plot_diurnal_cycle(df, var, f"../figures/notebooks/01_diurnal_{var}")

# %%
plot_correlation_matrix(df, "../figures/notebooks/01_corr")

# %% [markdown]
# ## 2019 haze episode close-up
#
# Southeast Asia experienced a severe transboundary haze event in September 2019.
# We highlight hourly PM2.5 and PM10 during August-October 2019.

# %%
import matplotlib.pyplot as plt

mask = (df["datetime"] >= "2019-08-01") & (df["datetime"] <= "2019-10-31")
episode = df.loc[mask]
if not episode.empty:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(episode["datetime"], episode["pm25"], label="PM2.5", linewidth=0.8)
    ax.plot(episode["datetime"], episode["pm10"], label="PM10", linewidth=0.8, alpha=0.7)
    ax.set_title("2019 Southeast Asian haze episode - Seberang Jaya")
    ax.set_ylabel("ug/m3")
    ax.legend()
    fig.savefig("../figures/notebooks/01_haze2019.png", dpi=200, bbox_inches="tight")
