# Paper-to-code mapping

This document traces each manuscript artefact back to the pipeline artefact
and the plotting script that produces it. Reviewers can use this table to
verify reproducibility.

## How the manuscript artefacts are produced

All quantitative results in the manuscript derive from the single
orchestrated run `make all`, which executes:

1. `make data` — raw Excel ingestion, quality control, feature engineering,
   and walk-forward partitioning. Outputs into `data/processed/`.
2. `make train` — training of the baselines (persistence, XGBoost, LSTM)
   and the Atmospheric Intelligence Engine (AIE). Trained-model checkpoints
   and training histories are saved under `artifacts/default/models/`.
3. `make evaluate` — per-horizon metrics, calibration and reliability
   diagnostics, and per-model prediction dumps. Saved under
   `artifacts/default/metrics.csv`, `artifacts/default/uq.npz`, and
   `artifacts/default/predictions/`.
4. `make scenarios` — exploratory driver-sensitivity scenario analysis.
   Saved as `artifacts/default/scenarios.csv`.

Figures and tables are then regenerated from those artefacts by the plotting
scripts under `scripts/`.

## Pipeline artefacts consumed by each class of manuscript output

| Manuscript output class                           | Pipeline artefact                                                                |
|---------------------------------------------------|----------------------------------------------------------------------------------|
| Study site and data description                   | `data/processed/seberang_jaya_hourly.parquet`                                    |
| Training behaviour (loss curves)                  | `artifacts/default/models/*_history.json`                                        |
| Deterministic forecast skill (RMSE, MAE, R², IOA) | `artifacts/default/metrics.csv`                                                  |
| Probabilistic forecast skill (coverage, CRPS)     | `artifacts/default/uq.npz`                                                       |
| Model predictions for scatter and reliability     | `artifacts/default/predictions/{persistence,xgboost,lstm,aie}.npz`               |
| Scenario results                                  | `artifacts/default/scenarios.csv`                                                |

## Scripts that regenerate figures and tables

| Script                       | Purpose                                                                                   |
|------------------------------|-------------------------------------------------------------------------------------------|
| `scripts/make_figures.py`    | Regenerate publication figures at 300 dpi PNG plus PDF vector, into `figures/generated/`. |
| `scripts/make_toc.py`        | Regenerate the table-of-contents graphic at 8 cm × 4 cm, in PNG, SVG, and PDF.            |

The per-figure and per-table captions in the manuscript identify the
generating script alongside each artefact. The captions are the
authoritative mapping and are kept in sync with the scripts at submission
time.

## Reproducibility protocol

```bash
conda env create -f environment.yml
conda activate aie
pip install -e ".[dev]"
make all          # data -> train -> evaluate -> scenarios
pytest -q
```

The run is deterministic given the fixed seeds in `configs/default.yaml`,
the pinned versions in `environment.yml`, and the training protocol
described in the Methods section of the manuscript. The same commit
reproduces the headline metrics bit-for-bit on the same hardware class.

## Reviewer notes

- The raw observational dataset is not redistributed with this repository
  (see `data/README.md` for the data-use agreement and access procedure).
  The small sample dataset under `data/sample/` is provided solely for
  continuous integration and smoke-testing and is explicitly marked as not
  suitable for scientific inference.
- The full trained model weights are retained under `artifacts/default/`
  when the pipeline is run locally. They are archived alongside the code
  in the Zenodo release for this version.
