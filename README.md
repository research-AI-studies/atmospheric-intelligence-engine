# Atmospheric Intelligence Engine (AIE)

[![CI](https://github.com/research-AI-studies/atmospheric-intelligence-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/research-AI-studies/atmospheric-intelligence-engine/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](LICENSE-data)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![DOI](https://zenodo.org/badge/1215887446.svg)](https://doi.org/10.5281/zenodo.19662860)

Reference implementation for the manuscript

> **Self-Optimizing Neural Architectures for Urban Air Quality Forecasting and
> Exploratory Scenario Analysis: A Case Study of Seberang Jaya, Malaysia
> (2018–2021).**
> 

This repository provides:

- An end-to-end, fully reproducible pipeline for **hourly air quality
  forecasting** at horizons 1 h – 168 h from a single monitoring station.
- A reference implementation of the **Atmospheric Intelligence Engine (AIE)**:
  a sequence-to-sequence neural architecture with MC-dropout based uncertainty
  quantification and lightweight online recalibration.
- Strong **baselines** (persistence, XGBoost, vanilla LSTM) for fair comparison.
- A **scenario sensitivity module** that propagates the trained model forward
  under perturbations of temperature and emission proxies.
- All scripts needed to regenerate every figure and table in the manuscript.

> **Study design.** The implementation targets the manuscript's experimental
> setting: hourly observations at the Seberang Jaya monitoring station
> (station code CA06P), Pulau Pinang, Malaysia, covering 2018–2021. Short-
> to medium-term forecast skill (1 h – 168 h) is evaluated by walk-forward
> validation on the 2021 test year; long-horizon behaviour is characterised
> by driver-perturbation scenario sensitivity analysis.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Quick start](#quick-start)
3. [Data](#data)
4. [Reproducing the paper](#reproducing-the-paper)
5. [Running with Docker](#running-with-docker)
6. [Development](#development)
7. [Citation](#citation)
8. [License](#license)
9. [Acknowledgements](#acknowledgements)

---

## Repository layout

```
atmospheric-intelligence-engine/
├── src/aie/              # Python package
│   ├── data/             # loading, QC, feature engineering, splits
│   ├── models/           # baselines, LSTM, AIE
│   ├── train.py          # training loop
│   ├── evaluate.py       # metrics & walk-forward evaluation
│   ├── uncertainty.py    # MC-dropout / calibration
│   ├── scenarios.py      # exploratory scenario analysis
│   └── plotting.py       # publication-ready figures
├── scripts/              # CLI entry points
├── configs/              # YAML experiment configs
├── notebooks/            # py:percent notebooks (one per figure group)
├── tests/                # pytest suite
├── data/sample/          # tiny synthetic slice for CI (raw data NOT included)
├── figures/              # generated figures (git-ignored except placeholders)
├── docs/                 # paper-to-code mapping
├── .github/workflows/    # CI configuration
├── environment.yml       # conda environment (Alibaba Cloud mirror)
├── requirements.txt      # pip fallback
├── pyproject.toml        # build metadata
├── Dockerfile
├── Makefile
└── PUBLISH.md            # GitHub + Zenodo release procedure
```

---

## Quick start

### 1. Create the conda environment (Alibaba Cloud mirror)

```bash
conda env create -f environment.yml
conda activate aie
```

### 2. (Optional) Install as an editable package

```bash
pip install -e ".[dev]"
```

### 3. Smoke-test on the bundled synthetic sample

```bash
make smoke
```

This runs the full pipeline end-to-end on the tiny synthetic dataset shipped in
`data/sample/`. It takes < 2 minutes on CPU.

### 4. Run the real pipeline

Place the real Excel file at:

```
data/raw/Seberang Jaya, Pulau Pinang_AIR QUALITY 2018-2021.xlsx
```

Then:

```bash
make data          # ingest + QC
make train         # train all configured models
make evaluate      # metrics + calibration + forecast-skill curves
make scenarios     # exploratory long-horizon scenarios
make figures       # regenerate every figure used in the manuscript
```

---

## Data

### Real observational data (not redistributed)

The primary dataset consists of hourly observations of PM10, PM2.5, SO₂, NO₂,
O₃ (2018–2020 only), CO, wind direction, wind speed, relative humidity, and
ambient temperature at the Seberang Jaya continuous air quality monitoring
station (station code **CA06P**), for the period **1 January 2018 – 31 December
2021** (35 063 hourly records across four annual sheets).

The raw file is **not redistributed** in this repository because it was
provided under a research-use agreement. Requests for the raw data should be
directed to the corresponding author.

### Synthetic sample (for CI and first-time users)

A small synthetic slice with the same schema is provided in `data/sample/` so
that the full pipeline can be exercised without access to the real data. It is
generated reproducibly by
[`scripts/generate_synthetic_sample.py`](scripts/generate_synthetic_sample.py)
from a fixed random seed; it is **not** suitable for scientific inference.

---

## Reproducing the paper

Every figure and table in the manuscript is produced by a deterministic script.
The full mapping is in [`docs/paper_to_code_mapping.md`](docs/paper_to_code_mapping.md).

Short version:

| Manuscript artefact                           | Command                                |
|-----------------------------------------------|----------------------------------------|
| Figures 1–2 (study area, data overview)       | `python scripts/make_figures.py eda`   |
| Figure 3 (architecture diagram)               | `python scripts/make_figures.py arch`  |
| Figures 4–6 (forecast-skill curves)           | `python scripts/make_figures.py skill` |
| Figure 7 (calibration / uncertainty)          | `python scripts/make_figures.py uq`    |
| Figures 8–9 (scenario sensitivity)            | `python scripts/make_figures.py scn`   |
| Table 1 (metrics summary)                     | `python scripts/make_figures.py table` |
| Table-of-contents graphic (8 cm × 4 cm)       | `python scripts/make_toc.py`           |

All outputs are written to `figures/` (PNG 300 dpi + PDF/SVG vector).

---

## Running with Docker

```bash
docker build -t aie:latest .
docker run --rm -it -v "$PWD:/work" aie:latest make smoke
```

A GPU variant is available via `docker compose up aie-gpu` (requires the
NVIDIA Container Toolkit).

---

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest -q
ruff check .
mypy src/aie
```

CI runs the full test suite plus a smoke pipeline on every push and pull
request via [GitHub Actions](.github/workflows/ci.yml).

---

## Citation

If you use this code, please cite both the software and the manuscript:

```bibtex
@software{aie_software_2026,
  author       = {Huang, Hai and Chen, Tet-Khuan and Lai, Ngan-Kuen and Lv, Wei and Liang, Zhan-Fan},
  title        = {Atmospheric Intelligence Engine (AIE): reference implementation},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19662860},
  url          = {https://github.com/research-AI-studies/atmospheric-intelligence-engine}
}
```

See [`CITATION.cff`](CITATION.cff) for the machine-readable form.

---

## License

- **Source code**: [MIT](LICENSE)
- **Figures, derived data, and documentation**: [CC BY 4.0](LICENSE-data)

The raw observational dataset is **not** released under these licenses and is
governed by the provider's data-use agreement; see [Data](#data).

---

## Acknowledgements

The authors thank the Department of Environment Malaysia (DOE / Jabatan Alam
Sekitar) for providing the monitoring-station data used in this study.
Computational resources and funding acknowledgements are listed in the
manuscript.
