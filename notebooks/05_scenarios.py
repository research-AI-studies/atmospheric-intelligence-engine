# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 05. Scenario sensitivity analysis
#
# Sensitivity of the trained AIE model to perturbations of temperature and
# emission scaling, driven by block-bootstrapped historical meteorology.

# %%
from pathlib import Path

import torch

from aie.config import load_config
from aie.data import build_features
from aie.plotting import apply_style, plot_scenarios
from aie.scenarios import run_scenarios
from aie.utils import resolve_device

apply_style()
cfg = load_config("../configs/default.yaml")
device = resolve_device(cfg.device)

# %%
import pandas as pd

features = pd.read_parquet("../artifacts/default/features.parquet")

# Reload the trained AIE model
mcfg = next(m for m in cfg.models if m.name == "aie")

from aie.models.aie import AtmosphericIntelligenceEngine

feature_cols = [c for c in features.columns if c not in {"datetime", "target"}]
model = AtmosphericIntelligenceEngine(
    n_features=len(feature_cols),
    n_horizons=len(mcfg.horizons),
    hidden_size=mcfg.hidden_size,
    num_transformer_layers=mcfg.num_layers,
    dropout=mcfg.dropout,
)
state = torch.load("../artifacts/default/models/aie.pt", map_location=device)
model.load_state_dict(state["state_dict"])
model = model.to(device).eval()

# %%
results = run_scenarios(
    model=model,
    feature_df=features,
    feature_cols=feature_cols,
    horizons=mcfg.horizons,
    input_window=mcfg.input_window,
    temperature_deltas=cfg.scenarios.temperature_delta_c,
    emission_scales=cfg.scenarios.emission_scale,
    horizon_hours=cfg.scenarios.horizon_hours,
    n_members=cfg.scenarios.n_members,
    device=device,
)

plot_scenarios(results, Path("../figures/notebooks/05_scenarios"))
