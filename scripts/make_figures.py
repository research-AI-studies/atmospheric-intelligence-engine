"""Regenerate publication figures from saved pipeline artefacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from aie.data import load_raw_excel
from aie.plotting import (
    apply_style,
    plot_correlation_matrix,
    plot_diurnal_cycle,
    plot_missingness_heatmap,
    plot_predicted_vs_observed,
    plot_reliability_diagram,
    plot_skill_vs_horizon,
)

FIG_DIR = Path("figures/generated")


def _load_processed(artifacts_dir: Path) -> pd.DataFrame:
    path = Path("data/processed/seberang_jaya_hourly.parquet")
    if path.exists():
        return pd.read_parquet(path)
    smoke_path = Path("data/processed/smoke_hourly.parquet")
    if smoke_path.exists():
        return pd.read_parquet(smoke_path)
    # Fall back to the raw file if no processed parquet is available.
    raw = Path("data/raw/Seberang Jaya, Pulau Pinang_AIR QUALITY 2018-2021.xlsx")
    if raw.exists():
        return load_raw_excel(raw)
    raise FileNotFoundError("No processed or raw data available. Run `make data` first.")


def figures_eda(artifacts_dir: Path) -> None:
    df = _load_processed(artifacts_dir)
    plot_missingness_heatmap(df, FIG_DIR / "fig01_missingness")
    for var in ("pm25", "pm10", "o3", "no2"):
        if var in df.columns:
            plot_diurnal_cycle(df, var, FIG_DIR / f"fig02_diurnal_{var}")
    plot_correlation_matrix(df, FIG_DIR / "fig03_correlation")


def figures_skill(artifacts_dir: Path) -> None:
    metrics_path = artifacts_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"[WARN] {metrics_path} missing - run `make evaluate` first.")
        return
    metrics = pd.read_csv(metrics_path)
    by_model: dict[str, pd.DataFrame] = {name: g for name, g in metrics.groupby("model")}
    for metric in ("rmse", "mae", "r2", "ioa"):
        plot_skill_vs_horizon(by_model, metric, FIG_DIR / f"fig04_skill_{metric}")

    preds_dir = artifacts_dir / "predictions"
    if preds_dir.exists():
        for npz_path in preds_dir.glob("*.npz"):
            name = npz_path.stem
            data = np.load(npz_path)
            horizons = data["horizons"].tolist()
            for j, h in enumerate(horizons):
                plot_predicted_vs_observed(
                    data["y_true"][:, j],
                    data["y_pred"][:, j],
                    int(h),
                    FIG_DIR / f"fig05_{name}_h{int(h)}_scatter",
                )


def figures_uq(artifacts_dir: Path) -> None:
    path = artifacts_dir / "uq.npz"
    if not path.exists():
        print(f"[WARN] {path} missing - run the AIE model evaluation first.")
        return
    data = np.load(path)
    plot_reliability_diagram(data["coverage"], data["levels"], FIG_DIR / "fig06_reliability")


def figures_table(artifacts_dir: Path) -> None:
    metrics_path = artifacts_dir / "metrics.csv"
    if not metrics_path.exists():
        print(f"[WARN] {metrics_path} missing - run `make evaluate` first.")
        return
    metrics = pd.read_csv(metrics_path)
    out = FIG_DIR / "table1_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics.round(3).to_csv(out, index=False)
    print(f"Wrote {out}")


def figures_arch(artifacts_dir: Path) -> None:
    """Emit a minimal placeholder architecture diagram (vector)."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    boxes = [
        ("Input (T x F)", 0.02, 0.4, 0.17),
        ("Projection", 0.22, 0.4, 0.11),
        ("TCN blocks\n(dilations 1,2,4)", 0.36, 0.45, 0.15),
        ("Transformer\nencoder", 0.54, 0.45, 0.14),
        ("GRN head", 0.71, 0.4, 0.11),
        ("Horizons", 0.85, 0.4, 0.12),
    ]
    for label, x, y, w in boxes:
        ax.add_patch(plt.Rectangle((x, y - 0.15), w, 0.3, fill=False))
        ax.text(x + w / 2, y, label, ha="center", va="center", fontsize=9)
    for (_, x1, y1, w1), (_, x2, y2, _) in zip(boxes[:-1], boxes[1:], strict=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1 + w1, y1), arrowprops={"arrowstyle": "->"})
    ax.axis("off")
    fig.suptitle("Atmospheric Intelligence Engine - architecture schematic")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_arch.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "fig_arch.pdf", bbox_inches="tight")
    print("Wrote architecture schematic.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "target",
        nargs="?",
        default="all",
        choices=["eda", "arch", "skill", "uq", "scn", "table", "all"],
    )
    parser.add_argument("--artifacts", default="artifacts/default", type=Path)
    args = parser.parse_args()

    apply_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.target in ("eda", "all"):
        figures_eda(args.artifacts)
    if args.target in ("arch", "all"):
        figures_arch(args.artifacts)
    if args.target in ("skill", "all"):
        figures_skill(args.artifacts)
    if args.target in ("uq", "all"):
        figures_uq(args.artifacts)
    if args.target in ("table", "all"):
        figures_table(args.artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
