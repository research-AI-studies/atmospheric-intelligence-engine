# Paper-to-code mapping

This document traces each manuscript artefact (figure, table, or textual
claim) back to the exact script, module, or notebook that produces it.
Reviewers can use this table to verify reproducibility.

## Figures

| Manuscript | Script / notebook | Output file(s) |
|------------|-------------------|----------------|
| Fig. 1 — Study area map           | Manually prepared (outside the code repo) | `figures/static/fig_study_area.pdf` |
| Fig. 2 — Data availability heatmap | `scripts/make_figures.py eda`            | `figures/generated/fig01_missingness.{png,pdf}` |
| Fig. 3 — Diurnal and seasonal cycles | `scripts/make_figures.py eda`          | `figures/generated/fig02_diurnal_*.{png,pdf}` |
| Fig. 4 — Correlation matrix        | `scripts/make_figures.py eda`           | `figures/generated/fig03_correlation.{png,pdf}` |
| Fig. 5 — Architecture schematic    | `scripts/make_figures.py arch`          | `figures/generated/fig_arch.{png,pdf}` |
| Fig. 6 — Forecast skill vs. horizon (RMSE) | `scripts/make_figures.py skill` | `figures/generated/fig04_skill_rmse.{png,pdf}` |
| Fig. 7 — Predicted vs. observed (AIE, h = 24) | `scripts/make_figures.py skill` | `figures/generated/fig05_aie_h24_scatter.{png,pdf}` |
| Fig. 8 — Reliability diagram (MC-dropout) | `scripts/make_figures.py uq`        | `figures/generated/fig06_reliability.{png,pdf}` |
| Fig. 9 — Exploratory scenarios     | `scripts/run_pipeline.py --stage scenarios` + `notebooks/05_scenarios.py` | `figures/notebooks/05_scenarios.{png,pdf}` |
| TOC graphic (8 cm × 4 cm)          | `scripts/make_toc.py`                   | `figures/generated/toc_graphic.{png,pdf}` |

## Tables

| Manuscript | Source |
|------------|--------|
| Table 1 — Summary of observational dataset | `src/aie/data/loader.py` (`load_raw_excel`) + `notebooks/01_eda.py` |
| Table 2 — Per-horizon metrics              | `artifacts/default/metrics.csv` (via `scripts/make_figures.py table`) |
| Table 3 — Calibration scores (coverage, CRPS) | `artifacts/default/uq.npz` + `notebooks/04_uncertainty.py` |

## Key textual claims

| Claim | Where it is computed |
|-------|----------------------|
| Dataset size (≈ 35,000 hourly records) | `aie.data.loader.load_raw_excel` (reports the final count in the log) |
| Per-horizon skill of the AIE model     | `artifacts/default/metrics.csv` rows where `model == "aie"` |
| Uncertainty calibration                | `artifacts/default/uq.npz` (`coverage` at nominal `levels`) |
| Scenario sensitivity methodology       | Module docstring at the top of `aie/scenarios.py` and notebook 05 |

## Reproducibility protocol

```bash
conda env create -f environment.yml
conda activate aie
pip install -e ".[dev]"
make all          # data -> train -> evaluate -> scenarios -> figures -> toc
pytest -q
```

The `make all` target is deterministic given the fixed seeds in
`configs/default.yaml`; the same commit should reproduce the figures and
tables bit-for-bit on the same hardware class.
