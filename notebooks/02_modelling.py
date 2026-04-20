# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02. Model training
#
# Train all four models (persistence, XGBoost, LSTM, AIE) on the walk-forward
# train split and inspect their learning curves.

# %%
from aie.config import load_config
from aie.pipeline import Pipeline

cfg = load_config("../configs/default.yaml")
pipeline = Pipeline(cfg)

# %%
features = pipeline.run_data()
features.head()

# %%
models = pipeline.run_train()
list(models)
