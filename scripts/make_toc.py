"""Generate the RSC Table-of-Contents graphic.

RSC specification: maximum 8 cm wide x 4 cm high. This script renders at
exactly that size at 600 dpi (PNG) and also emits vector PDF and SVG.

Layout
------
Left panel (~58% width): multi-horizon forecast illustration
    - observed PM2.5 (blue)
    - AIE forecast (red) with dashed forecast region
    - 95% MC-dropout uncertainty band
    - dotted vertical markers at the evaluation horizons
Right panel (~42% width): compact vertical architecture stack
    Input -> TCN -> Transformer -> MC-dropout head -> horizons
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path("figures/generated")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54
WIDTH_CM = 8.0
HEIGHT_CM = 4.0

# RSC-friendly, colour-blind-safe palette
OBS_COLOR = "#1f4e79"
PRED_COLOR = "#c0392b"
BAND_COLOR = "#c0392b"
BOX_EDGE = "#2c3e50"
BOX_FILL = "#ecf0f1"
ACCENT = "#2471a3"


def _left_panel(ax: plt.Axes) -> None:
    rng = np.random.default_rng(7)
    t_obs = np.arange(0, 120)
    t_fc = np.arange(120, 200)

    trend = 18 + 10 * np.sin(np.arange(200) / 14)
    obs = trend[:120] + rng.normal(0, 1.2, size=t_obs.size)
    fc = trend[120:] + rng.normal(0, 0.5, size=t_fc.size)
    band = 1.0 + 0.03 * (t_fc - t_fc[0])  # uncertainty widens with horizon

    ax.plot(t_obs, obs, color=OBS_COLOR, linewidth=0.9, solid_capstyle="round")
    ax.plot(t_fc, fc, color=PRED_COLOR, linewidth=0.9, linestyle="--")
    ax.fill_between(
        t_fc, fc - 1.96 * band, fc + 1.96 * band, color=BAND_COLOR, alpha=0.18, linewidth=0
    )

    ax.axvline(120, color="#7f8c8d", linewidth=0.4, linestyle=":")
    ax.text(120, 2.5, "now", fontsize=4.5, ha="center", color="#7f8c8d")

    ax.text(55, 38, "observed", fontsize=5, color=OBS_COLOR, ha="center")
    ax.text(160, 38, "AIE forecast  95% CI", fontsize=5, color=PRED_COLOR, ha="center")

    ax.set_xlim(0, 199)
    ax.set_ylim(0, 42)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("time (hours)", fontsize=5, labelpad=1)
    ax.set_ylabel("PM$_{2.5}$", fontsize=5, labelpad=1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.4)
        ax.spines[sp].set_color("#7f8c8d")


def _add_box(
    ax: plt.Axes, x: float, y: float, w: float, h: float, label: str
) -> tuple[float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.005,rounding_size=0.015",
        linewidth=0.5,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FILL,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=5, color=BOX_EDGE)
    return (x + w / 2, y)  # bottom-middle anchor


def _right_panel(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box_w, box_h = 0.78, 0.15
    x = (1 - box_w) / 2
    ys = [0.80, 0.61, 0.42, 0.23]
    labels = ["Input features", "Dilated TCN", "Transformer", "MC-dropout head"]
    anchors = []
    for y, label in zip(ys, labels, strict=True):
        anchors.append(_add_box(ax, x, y, box_w, box_h, label))

    for i in range(len(anchors) - 1):
        cx, cy_bottom = anchors[i]
        cy_top_next = ys[i + 1] + box_h
        arrow = FancyArrowPatch(
            (cx, cy_bottom),
            (cx, cy_top_next),
            arrowstyle="-|>",
            mutation_scale=4,
            linewidth=0.5,
            color=BOX_EDGE,
        )
        ax.add_patch(arrow)

    cx = anchors[-1][0]
    top = ys[-1]
    ax.annotate(
        "h = 1, 6, 24, 72, 168 h",
        xy=(cx, 0.09),
        ha="center",
        va="center",
        fontsize=4.5,
        color=ACCENT,
        weight="bold",
    )
    arrow = FancyArrowPatch(
        (cx, top),
        (cx, 0.14),
        arrowstyle="-|>",
        mutation_scale=4,
        linewidth=0.5,
        color=BOX_EDGE,
    )
    ax.add_patch(arrow)


def make_toc() -> None:
    fig = plt.figure(figsize=(WIDTH_CM * CM, HEIGHT_CM * CM), dpi=600)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.4, 1.0], wspace=0.08, left=0.06, right=0.98, top=0.92, bottom=0.10
    )

    ax_left = fig.add_subplot(gs[0])
    _left_panel(ax_left)

    ax_right = fig.add_subplot(gs[1])
    _right_panel(ax_right)

    fig.suptitle(
        "Atmospheric Intelligence Engine - multi-horizon urban air quality forecasting",
        fontsize=5.2,
        y=0.985,
        color=BOX_EDGE,
    )

    out = FIG_DIR / "toc_graphic"
    fig.savefig(out.with_suffix(".png"), dpi=600, pad_inches=0.02)
    fig.savefig(out.with_suffix(".pdf"), pad_inches=0.02)
    fig.savefig(out.with_suffix(".svg"), pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')}, .pdf, and .svg (exact {WIDTH_CM} x {HEIGHT_CM} cm).")


if __name__ == "__main__":
    make_toc()
