"""Raw Excel ingestion for the Seberang Jaya 2018-2021 dataset.

The source workbook contains one sheet per calendar year (``2018``, ``2019``,
``2020``, ``2021``). Sheet layouts differ slightly:

* 2018, 2020, 2021 have a blank row then the header row ("STATION ID",
  "LOCATION", "DATE TIME", ...).
* 2019 starts with the header on the first row.
* 2021 does NOT contain an O3 column.

This loader auto-detects the header row on each sheet, normalises column
names to ``snake_case``, concatenates everything into a tz-naive hourly
``pandas.DataFrame``, and returns it sorted by timestamp.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Canonical column schema used throughout the repository.
CANONICAL_COLUMNS: list[str] = [
    "station_id",
    "location",
    "datetime",
    "pm10",
    "pm25",
    "so2",
    "no2",
    "o3",
    "co",
    "wind_direction",
    "wind_speed",
    "relative_humidity",
    "temperature",
]

# Regex mapping from messy raw headers (after stripping units / symbols) to
# canonical names. Order matters; the first match wins.
_HEADER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"station\s*id", re.I), "station_id"),
    (re.compile(r"location", re.I), "location"),
    (re.compile(r"date\s*time", re.I), "datetime"),
    (re.compile(r"pm\s*2\.?5", re.I), "pm25"),
    (re.compile(r"pm\s*10", re.I), "pm10"),
    (re.compile(r"so\s*2", re.I), "so2"),
    (re.compile(r"no\s*2", re.I), "no2"),
    (re.compile(r"\bo\s*3\b", re.I), "o3"),
    (re.compile(r"\bco\b", re.I), "co"),
    (re.compile(r"wind\s*direction", re.I), "wind_direction"),
    (re.compile(r"wind\s*speed", re.I), "wind_speed"),
    (re.compile(r"relative\s*humidity", re.I), "relative_humidity"),
    (re.compile(r"(ambient\s*)?temperature", re.I), "temperature"),
]


def _match_canonical(raw_header: str) -> str | None:
    for pattern, canonical in _HEADER_PATTERNS:
        if pattern.search(raw_header):
            return canonical
    return None


def _detect_header_row(sheet_df: pd.DataFrame, max_scan_rows: int = 6) -> int:
    """Return the row index that contains the column headers."""

    for i in range(min(max_scan_rows, len(sheet_df))):
        row = sheet_df.iloc[i].astype(str).tolist()
        if any("DATE TIME" in str(v).upper() for v in row):
            return i
    raise ValueError("Could not locate a header row containing 'DATE TIME'.")


def _load_single_sheet(excel: pd.ExcelFile, sheet: str) -> pd.DataFrame:
    probe = excel.parse(sheet_name=sheet, header=None, nrows=6)
    header_row = _detect_header_row(probe)
    df = excel.parse(sheet_name=sheet, header=header_row)
    df = df.dropna(how="all").copy()

    rename: dict[str, str] = {}
    for col in df.columns:
        canonical = _match_canonical(str(col))
        if canonical is not None:
            rename[col] = canonical
    df = df.rename(columns=rename)

    # Drop any unmatched columns.
    df = df[[c for c in df.columns if c in CANONICAL_COLUMNS]]

    # Ensure every canonical column exists (O3 may be absent in 2021).
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[CANONICAL_COLUMNS]
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    numeric_cols = [c for c in CANONICAL_COLUMNS if c not in {"station_id", "location", "datetime"}]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["sheet_year"] = int(sheet)
    return df


def load_raw_excel(path: str | Path) -> pd.DataFrame:
    """Load and concatenate all per-year sheets into one hourly DataFrame."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Raw Excel file not found: {p}")
    logger.info("Reading %s", p)

    excel = pd.ExcelFile(p)
    frames = []
    for sheet in sorted(excel.sheet_names):
        if not sheet.isdigit():
            continue
        logger.info("  - sheet %s", sheet)
        frames.append(_load_single_sheet(excel, sheet))
    if not frames:
        raise RuntimeError("No per-year sheets were found in the workbook.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)

    # Reindex onto a strict hourly grid so downstream code can rely on it.
    full_index = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="H")
    df = df.set_index("datetime").reindex(full_index)
    df.index.name = "datetime"
    df = df.reset_index()

    # Forward-fill the two object columns so they remain constant.
    for col in ("station_id", "location"):
        df[col] = df[col].ffill().bfill()

    logger.info("Loaded %d hourly rows from %s to %s", len(df), df["datetime"].min(), df["datetime"].max())
    return df


def save_processed(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist the cleaned DataFrame as Parquet."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    logger.info("Wrote %d rows to %s", len(df), p)
    return p
