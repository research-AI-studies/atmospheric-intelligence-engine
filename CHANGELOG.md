# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-20

### Added
- Initial public release accompanying the first submission of the manuscript
  to *Environmental Science: Atmospheres*.
- `aie` Python package: data ingestion, quality control, feature engineering,
  walk-forward splits, baselines (persistence, XGBoost, LSTM), and the
  Atmospheric Intelligence Engine (AIE) architecture with MC-dropout.
- CLI scripts for data ingestion, training, evaluation, scenarios, and figure
  generation.
- Unit tests and a GitHub Actions CI workflow.
- Dockerfile and Makefile for fully reproducible execution.
- Synthetic sample dataset for smoke testing.
