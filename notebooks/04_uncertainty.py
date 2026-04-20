# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 04. Uncertainty quantification
#
# MC-dropout reliability diagram and CRPS for the AIE model.

# %%
from pathlib import Path

import numpy as np

from aie.plotting import apply_style, plot_reliability_diagram
from aie.uncertainty import crps_gaussian

apply_style()
data = np.load("../artifacts/default/uq.npz")

# %%
plot_reliability_diagram(data["coverage"], data["levels"], Path("../figures/notebooks/04_reliability"))

# %%
crps = crps_gaussian(data["y_true"].ravel(), data["y_mean"].ravel(), data["y_std"].ravel())
print(f"CRPS: {crps:.3f}")
