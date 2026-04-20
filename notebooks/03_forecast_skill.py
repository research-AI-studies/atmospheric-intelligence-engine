# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03. Forecast skill
#
# Load the evaluation artefacts and plot RMSE / MAE / R^2 / IOA per horizon.

# %%
from pathlib import Path

import pandas as pd

from aie.plotting import apply_style, plot_skill_vs_horizon

apply_style()
metrics = pd.read_csv("../artifacts/default/metrics.csv")
by_model = {name: g for name, g in metrics.groupby("model")}

# %%
for metric in ("rmse", "mae", "r2", "ioa"):
    plot_skill_vs_horizon(by_model, metric, Path(f"../figures/notebooks/03_skill_{metric}"))
