"""Generate Figures 1 and 2 for the paper.

Reads the canonical order-averaged verdict scores and writes fig1.pdf and
fig2.pdf. Every number plotted is recomputed here from verdict_scores_wide.csv
rather than read from a summary table, so the figures cannot drift from the
analysis. Printed values are echoed to stdout for cross-checking against the
Results tables.

Usage:  python analysis/make_figures.py [--outdir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results" / "canonical_v1"

BASELINE = "C1F1D1"

# Conditions that differ from the baseline in exactly one respect, plus the two
# judge substitutions. Order is fixed so both panels of Figure 1 share it.
CONTRASTS = [
    ("C2F1D1", "Criteria only", "instruction"),
    ("C1F2D1", "Format only", "instruction"),
    ("C1F1D2", "Scope only", "instruction"),
    ("L1_claude", "Judge: Claude", "judge"),
    ("L1_gemini", "Judge: Gemini", "judge"),
]

# The eight GPT-4.1 instruction conditions.
L2_CELLS = [
    "C1F1D1", "C1F1D2", "C1F2D1", "C1F2D2",
    "C2F1D1", "C2F1D2", "C2F2D1", "C2F2D2",
]

INK = "#1c1c1e"
MUTED = "#6b6f76"
INSTRUCTION = "#3a6ea5"
JUDGE = "#b5651d"
GRID = "#d8d8dc"

BOOTSTRAP = 2000
SEED = 20260829

# Boundary-change rate between two runs of the identical baseline specification.
# Read from the canonical output rather than duplicated as a literal figure value.
REPEAT_FLOOR = float(pd.read_csv(RESULTS / "repeat_floor.csv").loc[0, "boundary_change_pct"])
FLOOR = "#8a8a8f"


def style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def sign_class(s: pd.Series) -> pd.Series:
    """Sign class of an order-averaged score. NaN propagates."""
    return np.sign(s)


def paired_change(df: pd.DataFrame, cell: str) -> tuple[float, float, int]:
    """Paired aggregate change in percentage points, with its standard error."""
    pair = df[[BASELINE, cell]].dropna()
    d = 25.0 * (pair[cell] - pair[BASELINE])
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), len(d)


def boundary_rate(df: pd.DataFrame, cell: str, rng) -> tuple[float, float, float, int, int]:
    """Boundary-change rate (percent) with a case-bootstrap percentile interval."""
    pair = df[[BASELINE, cell]].dropna()
    changed = (sign_class(pair[cell]) != sign_class(pair[BASELINE])).to_numpy()
    n = len(changed)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    boot = 100.0 * changed[draws].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return 100.0 * changed.mean(), float(lo), float(hi), int(changed.sum()), n


def figure_one(df: pd.DataFrame, outdir: Path, rng) -> None:
    """Paired aggregate change beside boundary change, on a shared scale."""
    rows = []
    for cell, label, kind in CONTRASTS:
        delta, se, n_pair = paired_change(df, cell)
        rate, lo, hi, k, n_bnd = boundary_rate(df, cell, rng)
        rows.append(dict(label=label, kind=kind, delta=delta, se=se,
                         rate=rate, lo=lo, hi=hi, k=k, n=n_bnd))
        print(f"  {label:<16} paired {delta:+6.2f} pts (n={n_pair})   "
              f"boundary {rate:5.1f}% [{lo:.1f}, {hi:.1f}]  ({k}/{n_bnd})")

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(5.5, 2.05), sharey=True,
        gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1]},
    )
    y = np.arange(len(rows))[::-1]
    colours = [INSTRUCTION if r["kind"] == "instruction" else JUDGE for r in rows]
    limit = 50.0

    # Left: magnitude of the paired aggregate change, signed value annotated.
    ax_l.barh(y, [abs(r["delta"]) for r in rows], height=0.58,
              color=colours, alpha=0.85, linewidth=0)
    ax_l.errorbar([abs(r["delta"]) for r in rows], y,
                  xerr=[1.96 * r["se"] for r in rows],
                  fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6)
    for yi, r in zip(y, rows):
        ax_l.text(abs(r["delta"]) + 1.96 * r["se"] + 1.2, yi, f"{r['delta']:+.2f}",
                  va="center", ha="left", fontsize=7.2, color=INK)
    ax_l.set_xlim(0, limit)
    ax_l.set_xlabel("Change in aggregate score (points)")
    ax_l.set_title("Aggregate level", loc="left", color=INK)

    # Right: boundary-change rate on the same 0-50 span.
    ax_r.barh(y, [r["rate"] for r in rows], height=0.58,
              color=colours, alpha=0.85, linewidth=0)
    ax_r.errorbar([r["rate"] for r in rows], y,
                  xerr=np.array([[r["rate"] - r["lo"] for r in rows],
                                 [r["hi"] - r["rate"] for r in rows]]),
                  fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6)
    for yi, r in zip(y, rows):
        ax_r.text(r["hi"] + 1.2, yi, f"{r['rate']:.1f}%",
                  va="center", ha="left", fontsize=7.2, color=INK)
    ax_r.axvline(REPEAT_FLOOR, color=FLOOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax_r.text(REPEAT_FLOOR - 1.0, y.max() + 0.45, "repeat floor", ha="right", va="center",
              fontsize=6.8, color=FLOOR)
    ax_r.set_xlim(0, limit)
    ax_r.set_xlabel("Cases changing winner or tie (%)")
    ax_r.set_title("Case level", loc="left", color=INK)

    for ax in (ax_l, ax_r):
        ax.set_yticks(y)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax_l.set_yticklabels([r["label"] for r in rows])

    fig.savefig(outdir / "fig1.pdf")
    fig.savefig(outdir / "fig1.png")
    plt.close(fig)


def figure_two(df: pd.DataFrame, outdir: Path, rng) -> None:
    """Sensitivity count, and boundary change against baseline margin."""
    complete = df[L2_CELLS].dropna()
    base = sign_class(complete[BASELINE])
    others = [c for c in L2_CELLS if c != BASELINE]
    changed = pd.DataFrame(
        {c: (sign_class(complete[c]) != base).astype(int) for c in others}
    )
    counts = changed.sum(axis=1)
    n_complete = len(complete)
    unanimous = int((counts == 0).sum())
    print(f"  complete under all eight: {n_complete}")
    print(f"  unchanged under all seven: {unanimous} ({100*unanimous/n_complete:.1f}%)")

    margin = complete[BASELINE].abs()
    rate = 100.0 * changed.mean(axis=1)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(5.5, 1.95), gridspec_kw={"wspace": 0.32}
    )

    hist = counts.value_counts().reindex(range(8), fill_value=0)
    bars = ax_a.bar(hist.index, hist.to_numpy(), width=0.68,
                    color=INSTRUCTION, alpha=0.85, linewidth=0)
    bars[0].set_color(INK)
    for x, v in zip(hist.index, hist.to_numpy()):
        ax_a.text(x, v + 3, str(int(v)), ha="center", va="bottom", fontsize=7)
    ax_a.set_xlabel("Instruction conditions changing the outcome")
    ax_a.set_ylabel("Cases")
    ax_a.set_xticks(range(8))
    ax_a.set_ylim(0, hist.max() * 1.18)
    ax_a.set_title("Sensitivity across specifications", loc="left")

    levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    means, los, his, ns = [], [], [], []
    for m in levels:
        sel = rate[np.isclose(margin, m)].to_numpy()
        ns.append(len(sel))
        if len(sel) == 0:
            means.append(np.nan); los.append(np.nan); his.append(np.nan); continue
        draws = rng.integers(0, len(sel), size=(BOOTSTRAP, len(sel)))
        boot = sel[draws].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        means.append(sel.mean()); los.append(lo); his.append(hi)
        print(f"  margin {m:.1f}: {sel.mean():5.1f}% mean change rate  (n={len(sel)})")

    means = np.array(means)
    ax_b.errorbar(levels, means,
                  yerr=[means - np.array(los), np.array(his) - means],
                  fmt="o-", color=INSTRUCTION, ecolor=MUTED,
                  markersize=4, linewidth=1.2, elinewidth=0.8, capsize=2)
    for x, n in zip(levels, ns):
        ax_b.text(x, 3.5, f"n={n}", ha="center", fontsize=6.6, color=MUTED)
    ax_b.set_xlabel("Absolute baseline margin")
    ax_b.set_ylabel("Outcome changes (%)")
    ax_b.set_xticks(levels)
    ax_b.set_ylim(0, 100)
    ax_b.set_title("Change against decision margin", loc="left")

    for ax in (ax_a, ax_b):
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    fig.savefig(outdir / "fig2.pdf")
    fig.savefig(outdir / "fig2.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent.parent / "figures"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    style()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(RESULTS / "verdict_scores_wide.csv").set_index("uid")
    print(f"loaded {len(df)} cases from {RESULTS / 'verdict_scores_wide.csv'}")

    print("Figure 1:")
    figure_one(df, outdir, rng)
    print("Figure 2:")
    figure_two(df, outdir, rng)
    print(f"wrote fig1.pdf, fig2.pdf and .png previews to {outdir}")


if __name__ == "__main__":
    main()
