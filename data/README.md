# Data

## `data/raw/` (NOT distributed)

Place the original Excel workbook here:

```
data/raw/Seberang Jaya, Pulau Pinang_AIR QUALITY 2018-2021.xlsx
```

The raw file is **not** redistributed with this repository because it was
provided under a research-use agreement with the Department of Environment
Malaysia (DOE / Jabatan Alam Sekitar). Bona fide research requests should be
directed to the corresponding author.

## `data/sample/`

A reproducibly-generated synthetic replica of the workbook, for smoke-testing
and CI only. Not suitable for scientific inference. Regenerate with:

```
python scripts/generate_synthetic_sample.py --out data/sample
```

## `data/processed/` (generated)

Produced by the pipeline (`make data`). Git-ignored.

## `data/interim/` (generated)

Scratch space for intermediate artefacts. Git-ignored.
