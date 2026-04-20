# Notebooks

The notebooks below are stored in **py:percent** format
([jupytext](https://jupytext.readthedocs.io/)) so that they diff cleanly under
git and stay under the CI "no-binary" policy. Open them in JupyterLab/VSCode
and they will be rendered as notebooks automatically when `jupytext` is
installed.

To convert a single notebook to `.ipynb`:

```bash
jupytext --to notebook 01_eda.py
```

## Contents

| File | Purpose |
|------|---------|
| `01_eda.py`           | Exploratory data analysis: data availability, diurnal/seasonal cycles, correlations, 2019 haze episode close-up. |
| `02_modelling.py`     | Fit every configured model on a slice of the dataset and inspect training curves. |
| `03_forecast_skill.py`| Per-horizon RMSE / MAE / R² / IOA; predicted-vs-observed scatter. |
| `04_uncertainty.py`   | MC-dropout reliability diagram, CRPS, interval calibration. |
| `05_scenarios.py`     | Long-horizon scenario sensitivity under driver perturbations. |

Each notebook loads artefacts produced by `make data / train / evaluate` so
running the full Makefile once is enough to populate them.
