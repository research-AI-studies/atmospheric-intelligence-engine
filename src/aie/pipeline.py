"""Top-level orchestrator tying data, models, training, and evaluation together."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from aie.config import RunConfig, load_config
from aie.data import apply_qc, build_features, load_raw_excel, save_processed, walk_forward_split
from aie.dataset import SlidingWindowDataset
from aie.evaluate import per_horizon_metrics
from aie.models.aie import AtmosphericIntelligenceEngine
from aie.models.baselines import PersistenceModel, XGBoostForecaster
from aie.models.lstm import LSTMForecaster
from aie.train import train_model
from aie.uncertainty import mc_dropout_forecast, reliability_diagram
from aie.utils import configure_logging, ensure_dir, resolve_device, set_seed

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    processed_path: Path
    features_path: Path
    metrics_path: Path
    predictions_dir: Path
    models_dir: Path
    uq_path: Path


class Pipeline:
    """End-to-end pipeline.

    Stages (idempotent, can be run individually):

    1. ``data``      : ingest raw Excel, QC, feature engineering.
    2. ``train``     : fit every configured model on the train split.
    3. ``evaluate``  : walk-forward evaluation on the test split.
    4. ``scenarios`` : long-horizon sensitivity roll-outs under driver perturbations.
    """

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)
        self.out = ensure_dir(cfg.output_dir)
        self.art = PipelineArtifacts(
            processed_path=Path(cfg.data.processed_path),
            features_path=self.out / "features.parquet",
            metrics_path=self.out / "metrics.csv",
            predictions_dir=ensure_dir(self.out / "predictions"),
            models_dir=ensure_dir(self.out / "models"),
            uq_path=self.out / "uq.npz",
        )

    # --------------------------------------------------------------
    # Stage: data
    # --------------------------------------------------------------
    def run_data(self) -> pd.DataFrame:
        df = load_raw_excel(self.cfg.data.raw_path)
        df, reports = apply_qc(df, self.cfg.data)
        save_processed(df, self.art.processed_path)
        pd.DataFrame([r.as_dict() for r in reports]).to_csv(self.out / "qc_report.csv", index=False)

        features = build_features(df, self.cfg.features)
        features.to_parquet(self.art.features_path, index=False)
        logger.info("Stage 'data' complete: %d rows, %d features", *features.shape)
        return features

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def _load_features(self) -> pd.DataFrame:
        if not self.art.features_path.exists():
            return self.run_data()
        return pd.read_parquet(self.art.features_path)

    def _feature_columns(self, features: pd.DataFrame) -> list[str]:
        drop = {"datetime", "target"}
        return [c for c in features.columns if c not in drop]

    def _standardise(
        self, features: pd.DataFrame, split_mask: pd.Series
    ) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
        """Z-score standardise using statistics from the train split."""

        feature_cols = self._feature_columns(features)
        stats: dict[str, tuple[float, float]] = {}
        out = features.copy()
        for col in [*feature_cols, "target"]:
            values = out.loc[split_mask, col].astype(float)
            mu = float(np.nanmean(values))
            sd = float(np.nanstd(values))
            if not np.isfinite(sd) or sd < 1.0e-6:
                sd = 1.0
            stats[col] = (mu, sd)
            out[col] = (out[col].astype(float) - mu) / sd
        return out, stats

    # --------------------------------------------------------------
    # Stage: train
    # --------------------------------------------------------------
    def run_train(self) -> dict[str, object]:
        features = self._load_features()
        split = walk_forward_split(features, self.cfg.splits)
        logger.info("Split: %s", split.summary())

        standardised, stats = self._standardise(features, split.train)
        feature_cols = self._feature_columns(standardised)
        horizons = self.cfg.models[0].horizons  # horizons are shared across models

        models: dict[str, object] = {}
        for mcfg in self.cfg.models:
            logger.info("=== Training %s ===", mcfg.name)
            if mcfg.name == "persistence":
                models[mcfg.name] = PersistenceModel(horizons=horizons)
            elif mcfg.name == "xgboost":
                X_train = standardised.loc[split.train, feature_cols].reset_index(drop=True)
                y_train = standardised.loc[split.train, "target"].to_numpy(dtype=float)
                X_val = standardised.loc[split.val, feature_cols].reset_index(drop=True)
                y_val = standardised.loc[split.val, "target"].to_numpy(dtype=float)
                booster = XGBoostForecaster(
                    horizons=horizons,
                    n_estimators=mcfg.xgb_n_estimators,
                    max_depth=mcfg.xgb_max_depth,
                    random_state=mcfg.seed,
                )
                booster.fit(X_train, y_train, X_val, y_val)
                models[mcfg.name] = booster
            elif mcfg.name in {"lstm", "aie"}:
                train_ds = SlidingWindowDataset(
                    standardised.loc[split.train].reset_index(drop=True),
                    feature_cols,
                    horizons,
                    mcfg.input_window,
                )
                val_ds = SlidingWindowDataset(
                    standardised.loc[split.val].reset_index(drop=True),
                    feature_cols,
                    horizons,
                    mcfg.input_window,
                )
                train_loader = DataLoader(
                    train_ds, batch_size=mcfg.batch_size, shuffle=True, drop_last=False
                )
                val_loader = DataLoader(val_ds, batch_size=mcfg.batch_size, shuffle=False)
                if mcfg.name == "lstm":
                    model = LSTMForecaster(
                        n_features=len(feature_cols),
                        n_horizons=len(horizons),
                        hidden_size=mcfg.hidden_size,
                        num_layers=mcfg.num_layers,
                        dropout=mcfg.dropout,
                    )
                else:
                    model = AtmosphericIntelligenceEngine(
                        n_features=len(feature_cols),
                        n_horizons=len(horizons),
                        hidden_size=mcfg.hidden_size,
                        num_transformer_layers=mcfg.num_layers,
                        dropout=mcfg.dropout,
                    )
                model, history = train_model(model, train_loader, val_loader, mcfg, self.device)
                torch.save(
                    {"state_dict": model.state_dict(), "cfg": mcfg.__dict__},
                    self.art.models_dir / f"{mcfg.name}.pt",
                )
                with (self.art.models_dir / f"{mcfg.name}_history.json").open(
                    "w", encoding="utf-8"
                ) as fh:
                    json.dump(history.__dict__, fh, indent=2)
                models[mcfg.name] = model
            else:
                raise ValueError(f"Unknown model name: {mcfg.name}")

        with (self.art.models_dir / "feature_cols.json").open("w", encoding="utf-8") as fh:
            json.dump(feature_cols, fh)
        with (self.art.models_dir / "stats.pkl").open("wb") as fh:
            pickle.dump(stats, fh)
        logger.info("Stage 'train' complete")
        return models

    # --------------------------------------------------------------
    # Stage: evaluate
    # --------------------------------------------------------------
    def run_evaluate(self, models: dict[str, object] | None = None) -> pd.DataFrame:
        features = self._load_features()
        split = walk_forward_split(features, self.cfg.splits)
        standardised, stats = self._standardise(features, split.train)
        feature_cols = self._feature_columns(standardised)
        horizons = self.cfg.models[0].horizons

        if models is None:
            models = self._reload_models(feature_cols, horizons)

        target_mu, target_sd = stats["target"]
        metrics_frames: list[pd.DataFrame] = []

        for name, model in models.items():
            logger.info("Evaluating %s", name)
            if name == "persistence":
                y = standardised.loc[split.test, "target"].to_numpy(dtype=float)
                pred_std = model.predict(y)  # type: ignore[attr-defined]
                y_true = y.reshape(-1, 1) * target_sd + target_mu
                y_pred = pred_std * target_sd + target_mu
                mask = (~np.isnan(features.loc[split.test, "target"].to_numpy())).astype(np.float32)
                mask_mat = np.repeat(mask.reshape(-1, 1), len(horizons), axis=1)
                y_true_mat = np.repeat(y_true, len(horizons), axis=1)
            elif name == "xgboost":
                X = standardised.loc[split.test, feature_cols].reset_index(drop=True)
                pred_std = model.predict(X)  # type: ignore[attr-defined]
                y_std = standardised.loc[split.test, "target"].to_numpy(dtype=float)
                y_pred = pred_std * target_sd + target_mu
                y_true_mat = (
                    np.stack([np.roll(y_std, -h) for h in horizons], axis=1) * target_sd + target_mu
                )
                mask_mat = np.stack(
                    [np.roll((~np.isnan(y_std)).astype(np.float32), -h) for h in horizons], axis=1
                )
                mask_mat[-max(horizons) :, :] = 0
            else:
                y_true_mat, y_pred, mask_mat = self._torch_predict(
                    model,  # type: ignore[arg-type]
                    standardised.loc[split.test].reset_index(drop=True),
                    feature_cols,
                    horizons,
                    self.cfg.models[-1].input_window,
                    self.cfg.models[-1].batch_size,
                    stats,
                )

            mf = per_horizon_metrics(y_true_mat, y_pred, horizons, mask_mat)
            mf.insert(0, "model", name)
            metrics_frames.append(mf)
            np.savez_compressed(
                self.art.predictions_dir / f"{name}.npz",
                y_true=y_true_mat,
                y_pred=y_pred,
                mask=mask_mat,
                horizons=np.array(horizons),
            )

        metrics = pd.concat(metrics_frames, ignore_index=True)
        metrics.to_csv(self.art.metrics_path, index=False)

        # Uncertainty for the AIE model, if present
        if "aie" in models and isinstance(models["aie"], torch.nn.Module):
            self._run_uncertainty(models["aie"], standardised, split, feature_cols, horizons, stats)

        logger.info(
            "Stage 'evaluate' complete: %d rows written to %s", len(metrics), self.art.metrics_path
        )
        return metrics

    # --------------------------------------------------------------
    def _torch_predict(
        self,
        model: torch.nn.Module,
        frame: pd.DataFrame,
        feature_cols: list[str],
        horizons: list[int],
        input_window: int,
        batch_size: int,
        stats: dict[str, tuple[float, float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ds = SlidingWindowDataset(frame, feature_cols, horizons, input_window)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        model = model.to(self.device).eval()
        all_true, all_pred, all_mask = [], [], []
        with torch.no_grad():
            for x, y, m in loader:
                x = x.to(self.device)
                pred = model(x).cpu().numpy()
                all_true.append(y.numpy())
                all_pred.append(pred)
                all_mask.append(m.numpy())
        y_true_std = np.concatenate(all_true, axis=0)
        y_pred_std = np.concatenate(all_pred, axis=0)
        mask = np.concatenate(all_mask, axis=0)
        mu, sd = stats["target"]
        return y_true_std * sd + mu, y_pred_std * sd + mu, mask

    def _run_uncertainty(
        self,
        model: torch.nn.Module,
        standardised: pd.DataFrame,
        split,
        feature_cols: list[str],
        horizons: list[int],
        stats: dict[str, tuple[float, float]],
    ) -> None:
        mcfg = next(m for m in self.cfg.models if m.name == "aie")
        ds = SlidingWindowDataset(
            standardised.loc[split.test].reset_index(drop=True),
            feature_cols,
            horizons,
            mcfg.input_window,
        )
        loader = DataLoader(ds, batch_size=mcfg.batch_size, shuffle=False)
        y_true, y_mean, y_std, mask = mc_dropout_forecast(
            model, loader, self.device, n_samples=mcfg.mc_dropout_samples
        )
        mu, sd = stats["target"]
        y_true = y_true * sd + mu
        y_mean = y_mean * sd + mu
        y_std = y_std * sd
        levels = np.linspace(0.1, 0.95, 18)
        coverage = reliability_diagram(y_true.ravel(), y_mean.ravel(), y_std.ravel(), levels)
        np.savez_compressed(
            self.art.uq_path,
            y_true=y_true,
            y_mean=y_mean,
            y_std=y_std,
            mask=mask,
            levels=levels,
            coverage=coverage,
            horizons=np.array(horizons),
        )
        logger.info("Wrote UQ arrays to %s", self.art.uq_path)

    def _reload_models(self, feature_cols: list[str], horizons: list[int]) -> dict[str, object]:
        models: dict[str, object] = {}
        for mcfg in self.cfg.models:
            if mcfg.name == "persistence":
                models[mcfg.name] = PersistenceModel(horizons=horizons)
            elif mcfg.name == "xgboost":
                raise RuntimeError(
                    "XGBoost models must be retrained; no persistence layer provided."
                )
            else:
                path = self.art.models_dir / f"{mcfg.name}.pt"
                if not path.exists():
                    logger.warning("Skipping %s: checkpoint missing at %s", mcfg.name, path)
                    continue
                if mcfg.name == "lstm":
                    model = LSTMForecaster(
                        n_features=len(feature_cols),
                        n_horizons=len(horizons),
                        hidden_size=mcfg.hidden_size,
                        num_layers=mcfg.num_layers,
                        dropout=mcfg.dropout,
                    )
                else:
                    model = AtmosphericIntelligenceEngine(
                        n_features=len(feature_cols),
                        n_horizons=len(horizons),
                        hidden_size=mcfg.hidden_size,
                        num_transformer_layers=mcfg.num_layers,
                        dropout=mcfg.dropout,
                    )
                state = torch.load(path, map_location=self.device)
                model.load_state_dict(state["state_dict"])
                models[mcfg.name] = model.to(self.device)
        return models


def run_from_yaml(config_path: str, stage: str = "all") -> None:
    configure_logging()
    cfg = load_config(config_path)
    pipeline = Pipeline(cfg)
    if stage in ("data", "all"):
        pipeline.run_data()
    if stage in ("train", "all"):
        pipeline.run_train()
    if stage in ("evaluate", "all"):
        pipeline.run_evaluate()
