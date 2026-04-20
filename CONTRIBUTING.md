# Contributing

Thank you for your interest in this project. This repository primarily serves
as the reference implementation for a peer-reviewed manuscript; nevertheless,
issues and pull requests are welcome.

## Development setup

```bash
conda env create -f environment.yml
conda activate aie
pip install -e ".[dev]"
pre-commit install
```

## Style

- Python formatting: `black` (line length 100) + `ruff` (lint + isort).
- Type checking: `mypy` (non-strict).
- Docstrings: NumPy style.
- Commits: English only; follow the
  [Conventional Commits](https://www.conventionalcommits.org/) style where
  possible (e.g. `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `ci:`).

## Running tests

```bash
pytest -q
```

CI will also run lint (`ruff`, `black --check`) and a smoke pipeline.

## Reporting issues

When reporting an issue, please include:

- OS + Python version
- `conda list` output (or `pip freeze`)
- The exact command you ran and the full traceback
