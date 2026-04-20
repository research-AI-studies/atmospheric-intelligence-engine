"""Configuration dataclasses and YAML loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Configuration for raw data ingestion and QC."""

    raw_path: str = "data/raw/Seberang Jaya, Pulau Pinang_AIR QUALITY 2018-2021.xlsx"
    processed_path: str = "data/processed/seberang_jaya_hourly.parquet"
    station_id: str = "CA06P"
    location: str = "Seberang Jaya, Pulau Pinang"
    timezone: str = "Asia/Kuala_Lumpur"
    # Physically plausible ranges for QC (unit as in raw file).
    qc_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "pm10": (0.0, 1000.0),
            "pm25": (0.0, 800.0),
            "so2": (0.0, 0.5),
            "no2": (0.0, 0.5),
            "o3": (0.0, 0.3),
            "co": (0.0, 30.0),
            "wind_speed": (0.0, 40.0),
            "wind_direction": (0.0, 360.0),
            "relative_humidity": (0.0, 100.0),
            "temperature": (5.0, 45.0),
        }
    )
    max_gap_hours: int = 6  # max gap length eligible for interpolation


@dataclass
class FeatureConfig:
    """Feature engineering knobs."""

    target: str = "pm25"
    lag_hours: list[int] = field(default_factory=lambda: [1, 2, 3, 6, 12, 24, 48, 72, 168])
    rolling_windows: list[int] = field(default_factory=lambda: [3, 6, 24])
    include_calendar: bool = True
    include_wind_components: bool = True


@dataclass
class SplitConfig:
    """Walk-forward evaluation split."""

    train_years: list[int] = field(default_factory=lambda: [2018, 2019])
    val_years: list[int] = field(default_factory=lambda: [2020])
    test_years: list[int] = field(default_factory=lambda: [2021])


@dataclass
class ModelConfig:
    """Hyperparameters for a single model."""

    name: str = "aie"  # one of: persistence, xgboost, lstm, aie
    input_window: int = 168  # 7 days of hourly history
    horizons: list[int] = field(default_factory=lambda: [1, 6, 24, 72, 168])
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    mc_dropout_samples: int = 50
    batch_size: int = 128
    epochs: int = 30
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    patience: int = 5
    seed: int = 42
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 6


@dataclass
class ScenarioConfig:
    """Configuration for long-horizon scenario sensitivity analysis.

    Controls the roll-out length, the grid of driver perturbations, and the
    ensemble size used by :func:`aie.scenarios.run_scenarios`.
    """

    horizon_hours: int = 24 * 365  # one year of hourly roll-out
    temperature_delta_c: list[float] = field(default_factory=lambda: [0.0, +1.0, +2.0])
    emission_scale: list[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    n_members: int = 20


@dataclass
class RunConfig:
    """Top-level run configuration."""

    experiment_name: str = "default"
    output_dir: str = "artifacts"
    device: str = "auto"  # "cpu", "cuda", "auto"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    models: list[ModelConfig] = field(
        default_factory=lambda: [
            ModelConfig(name="persistence"),
            ModelConfig(name="xgboost"),
            ModelConfig(name="lstm"),
            ModelConfig(name="aie"),
        ]
    )
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)


def _model_from_dict(d: dict[str, Any]) -> ModelConfig:
    return ModelConfig(**d)


def load_config(path: str | Path) -> RunConfig:
    """Load a :class:`RunConfig` from a YAML file."""

    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    data = DataConfig(**raw.get("data", {}))
    features = FeatureConfig(**raw.get("features", {}))
    splits = SplitConfig(**raw.get("splits", {}))
    scenarios = ScenarioConfig(**raw.get("scenarios", {}))
    models_raw = raw.get("models")
    if models_raw:
        models = [_model_from_dict(m) for m in models_raw]
    else:
        models = RunConfig().models

    top = {k: v for k, v in raw.items() if k not in {"data", "features", "splits", "scenarios", "models"}}
    return RunConfig(
        **top,
        data=data,
        features=features,
        splits=splits,
        models=models,
        scenarios=scenarios,
    )
