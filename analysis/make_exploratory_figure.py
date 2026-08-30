"""Generate Figure 3, the prompt-level stability spectrum.

Figure 3 shows every prompt in the benchmark, shaded by how many of the nine
judge conditions change its winner-or-tie outcome. A stable core and a fully
unstable tail both exist.

Usage:  PYTHONPATH=analysis python analysis/make_exploratory_figure.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
INK, MUTED = "#1c1c1e", "#6b6f76"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
        "axes.labelcolor": INK, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def main() -> None:
    style()
    stability = pd.read_csv(HERE / "results" / "canonical_v1" / "stability_spectrum.csv")

    out = HERE / "figures"

    # ---- figure one: every prompt, shaded by how many conditions change it ----
    fig, ax_l = plt.subplots(figsize=(4.8, 3.4))
    counts = stability["change_count"].sort_values().to_numpy()

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "stab", ["#1f4e5f", "#4a90a4", "#9dc3d0", "#e8b04b", "#c0392b"])
    n0 = int((counts == 0).sum()); n9 = int((counts == 9).sum())

    # one square per prompt, drawn with a gap so individual items stay visible
    cols = 24
    rows = int(np.ceil(len(counts) / cols))
    for i, v in enumerate(counts):
        r, c = divmod(i, cols)
        ax_l.add_patch(mpl.patches.Rectangle((c, rows - 1 - r), 0.84, 0.84,
                                             facecolor=cmap(v / 9), linewidth=0))
    ax_l.set_xlim(-3.4, cols + 0.2)
    ax_l.set_ylim(-1.4, rows + 0.6)
    ax_l.set_aspect("equal")
    ax_l.set_xticks([]); ax_l.set_yticks([])
    for sp in ax_l.spines.values():
        sp.set_visible(False)

    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 9))
    cb = fig.colorbar(sm, ax=ax_l, orientation="horizontal", pad=0.03,
                      fraction=0.045, aspect=30)
    cb.set_ticks([0, 9])
    cb.ax.set_xticklabels(["0", "9"])
    cb.ax.tick_params(labelsize=7.5, color=MUTED, length=2)
    cb.outline.set_visible(False)
    # label the ends by what they mean, not by the count that produces them
    cb.ax.text(-0.02, 0.5, "stable", transform=cb.ax.transAxes, fontsize=8.5,
               color=INK, ha="right", va="center")
    cb.ax.text(1.02, 0.5, "unstable", transform=cb.ax.transAxes, fontsize=8.5,
               color=INK, ha="left", va="center")
    # the end labels and the 0-9 ticks say everything; the definition lives in
    # the caption rather than under the bar

    fig.savefig(out / "fig3.pdf")
    fig.savefig(out / "fig3.png")
    plt.close(fig)

    print(f"never change: {n0}   always change: {n9}   total plotted: {len(counts)}")
    print(f"wrote {out}/fig3.[pdf|png]")


if __name__ == "__main__":
    main()
