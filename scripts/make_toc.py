"""Generate the RSC Table-of-Contents graphic.

RSC specification: maximum 8 cm wide x 4 cm high. This script renders at
exactly that size at 600 dpi (PNG) and as vector PDF.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("figures/generated")
FIG_DIR.mkdir(parents=True, exist_ok=True)

CM = 1 / 2.54  # conversion inch <- cm
WIDTH_CM = 8.0
HEIGHT_CM = 4.0


def make_toc() -> None:
    fig = plt.figure(figsize=(WIDTH_CM * CM, HEIGHT_CM * CM), dpi=600)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.1)

    # Left panel: stylised multi-horizon forecast
    ax1 = fig.add_subplot(gs[0])
    rng = np.random.default_rng(0)
    t = np.arange(200)
    obs = 15 + 8 * np.sin(t / 12) + rng.normal(0, 1.5, size=t.size)
    pred = obs + rng.normal(0, 0.8, size=t.size)
    ax1.plot(t, obs, color="#1f77b4", linewidth=0.7, label="obs")
    ax1.plot(t, pred, color="#d62728", linewidth=0.7, label="AIE")
    ax1.fill_between(t, pred - 1.5, pred + 1.5, color="#d62728", alpha=0.2)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("multi-horizon forecast", fontsize=6)
    for sp in ax1.spines.values():
        sp.set_visible(False)

    # Right panel: stylised architecture
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    boxes = [
        (0.02, 0.45, "Input"),
        (0.24, 0.45, "TCN"),
        (0.44, 0.45, "Transformer"),
        (0.70, 0.45, "MC-dropout\nhead"),
    ]
    for x, y, label in boxes:
        ax2.add_patch(plt.Rectangle((x, y - 0.18), 0.18, 0.36, fill=False, linewidth=0.6))
        ax2.text(x + 0.09, y, label, ha="center", va="center", fontsize=5)
    for (x1, y1, _), (x2, y2, _) in zip(boxes[:-1], boxes[1:], strict=False):
        ax2.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1 + 0.18, y1),
            arrowprops={"arrowstyle": "->", "lw": 0.5},
        )
    ax2.set_title("Atmospheric Intelligence Engine", fontsize=6)

    out = FIG_DIR / "toc_graphic"
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    make_toc()
