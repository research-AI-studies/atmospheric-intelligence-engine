# ---------------------------------------------------------------------------
# Makefile for the Atmospheric Intelligence Engine (AIE).
# All targets are idempotent and can be run from a fresh clone.
# ---------------------------------------------------------------------------

PYTHON ?= python
CONFIG ?= configs/default.yaml
RAW    ?= data/raw/Seberang\ Jaya,\ Pulau\ Pinang_AIR\ QUALITY\ 2018-2021.xlsx

.PHONY: help env install lint test smoke sample data train evaluate scenarios figures toc all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

env:  ## Create the conda environment
	conda env create -f environment.yml

install:  ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Lint with ruff + black --check
	ruff check .
	black --check .

test:  ## Run unit tests
	pytest -q

sample:  ## Regenerate the synthetic sample dataset
	$(PYTHON) scripts/generate_synthetic_sample.py --out data/sample

smoke: sample  ## End-to-end pipeline on the synthetic sample (fast)
	$(PYTHON) scripts/run_pipeline.py --config configs/smoke.yaml

data:  ## Ingest + QC the real Excel file into data/processed/
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stage data

train:  ## Train all configured models
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stage train

evaluate:  ## Walk-forward evaluation, metrics, calibration
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stage evaluate

scenarios:  ## Long-horizon scenario sensitivity analysis
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG) --stage scenarios

figures:  ## Regenerate every figure used in the manuscript
	$(PYTHON) scripts/make_figures.py all

toc:  ## Render the table-of-contents graphic (8 cm x 4 cm)
	$(PYTHON) scripts/make_toc.py

all: data train evaluate scenarios figures toc  ## Full pipeline end-to-end

clean:  ## Remove generated artefacts (keeps raw data)
	rm -rf artifacts/ runs/ logs/ figures/generated/ data/processed/ data/interim/
