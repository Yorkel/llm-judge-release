"""October figures and table images for the camera-ready (for upload to Overleaf as PDFs).

Figures (same design as make_figures.py / make_exploratory_figure.py, October data):
  fig1.pdf  score change and % of prompts with a different winner, for the three single-choice setups,
            with the default setup's floor as a band (results/october_rerun/step2_table1_october.csv)
  fig2.pdf  left: how many of the 7 setups change each prompt; right: change rate by margin, setups
            against the default setup's rerun, margin from run 3 (step10_closeness_october.csv)
  fig3.pdf  one square per prompt, shaded by how many of the 7 setups change it (recomputed from data)
Tables (tabular part only; the caption stays in LaTeX):
  table1_main.pdf (excess with both floors), table1_full.pdf (both excess columns),
  table_conditions.pdf, table_thresholds.pdf, table_tally.pdf, table_margin.pdf
Writes to figures/october/, with .png previews. Reads results and data only.
Use --output-dir PATH to render a separate copy.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from october_step2_table1 import DEFAULT, SETUPS, runs

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results" / "october_rerun"
OUT = HERE / "figures" / "october"

INK, MUTED, INSTRUCTION, GRID, FLOOR = "#1c1c1e", "#6b6f76", "#3a6ea5", "#d8d8dc", "#8a8a8f"
SINGLE = [("C2F1D1", "Accuracy-first"), ("C1F2D1", "Label only"), ("C1F1D2", "+ description")]
ORDER = ["C1F1D1", "C1F1D2", "C1F2D1", "C1F2D2", "C2F1D1", "C2F1D2", "C2F2D1", "C2F2D2"]
LEVELS = {"C": {"1": "Arena-Hard", "2": "Accuracy"}, "F": {"1": "Reasoning", "2": "Label"},
          "D": {"1": "none", "2": "coding"}}


def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 8,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": INK,
        "text.color": INK, "axes.labelcolor": INK, "figure.dpi": 200,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def figure_one(t1):
    rows = [t1.set_index("cell").loc[c] for c, _ in SINGLE]
    floor = t1.set_index("cell").loc[DEFAULT]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(5.5, 1.55), sharey=True,
                                     gridspec_kw={"wspace": 0.08})
    y = np.arange(len(rows))[::-1]
    limit = 50.0
    crosses = [r.score_change_lo <= 0 <= r.score_change_hi for r in rows]
    mag = [abs(r.score_change) for r in rows]
    err = [[abs(r.score_change) - min(abs(r.score_change_lo), abs(r.score_change_hi)) if not z else abs(r.score_change)
            for r, z in zip(rows, crosses)],
           [max(abs(r.score_change_lo), abs(r.score_change_hi)) - abs(r.score_change) for r in rows]]
    ax_l.barh(y, mag, height=0.58, color=["none" if z else INSTRUCTION for z in crosses],
              edgecolor=INSTRUCTION, linewidth=[0.9 if z else 0 for z in crosses], alpha=0.85)
    ax_l.errorbar(mag, y, xerr=err, fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6)
    for yi, m, e, r in zip(y, mag, err[1], rows):
        ax_l.text(m + e + 1.2, yi, f"{r.score_change:+.2f}".replace("-", "\u2212"), va="center", ha="left", fontsize=7.2)
    ax_l.set_xlim(0, limit)
    ax_l.set_xlabel("Change in aggregate score (points)")
    ax_l.set_title("Aggregate level", loc="left")
    ax_l.text(limit - 1.0, y.max() + 0.3, "outlined bar:\ninterval includes zero", fontsize=6.8,
              color=MUTED, ha="right", va="top", linespacing=1.35)

    rate = [r.pct_different_winner for r in rows]
    ax_r.barh(y, rate, height=0.58, color=INSTRUCTION, alpha=0.85, linewidth=0)
    ax_r.errorbar(rate, y, xerr=[[r.pct_different_winner - r.pct_different_winner_lo for r in rows],
                                 [r.pct_different_winner_hi - r.pct_different_winner for r in rows]],
                  fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.6)
    for yi, r in zip(y, rows):
        ax_r.text(r.pct_different_winner_hi + 1.2, yi, f"{r.pct_different_winner:.1f}%", va="center",
                  ha="left", fontsize=7.2)
    ax_r.axvspan(floor.default_floor_lo, floor.default_floor_hi, color=FLOOR, alpha=0.16, linewidth=0, zorder=1)
    ax_r.axvline(floor.default_floor, color=FLOOR, linewidth=1.0, linestyle=(0, (4, 3)), zorder=3)
    ax_r.text(floor.default_floor_lo - 1.0, y.max() + 0.45, "default setup's floor", ha="right",
              va="center", fontsize=6.8, color=FLOOR)
    ax_r.set_xlim(0, limit)
    ax_r.set_xlabel("Prompts with a different winner (%)")
    ax_r.set_title("Prompt level", loc="left")
    for ax in (ax_l, ax_r):
        ax.set_yticks(y)
        ax.xaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    ax_l.set_yticklabels([name for _, name in SINGLE])
    save(fig, "fig1")


def figure_two(cl):
    hist = cl[cl.section == "C_setups_changing_prompt"].set_index("group")["n"]
    counts = hist.to_numpy()
    A = cl[cl.section == "A_by_margin"]
    levels = [0.0, 0.5, 1.0, 1.5, 2.0]
    get = lambda m, k: A[(A.group == f"margin {m:g}") & (A.measure == k)].iloc[0]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.5, 1.95), gridspec_kw={"wspace": 0.32})
    bars = ax_a.bar(range(8), counts, width=0.68, color=INSTRUCTION, alpha=0.85, linewidth=0)
    bars[0].set_color(INK)
    for x, v in enumerate(counts):
        ax_a.text(x, v + 3, str(int(v)), ha="center", va="bottom", fontsize=7)
    ax_a.set_xlabel("Setups changing the outcome")
    ax_a.set_ylabel("Prompts")
    ax_a.set_xticks(range(8))
    ax_a.set_ylim(0, counts.max() * 1.18)
    ax_a.set_title("Sensitivity across setups", loc="left")

    sc = np.array([get(m, "setup_change").value for m in levels])
    lo = np.array([get(m, "setup_change").lo for m in levels])
    hi = np.array([get(m, "setup_change").hi for m in levels])
    rr = np.array([get(m, "default_rerun").value for m in levels])
    ns = [int(get(m, "setup_change").n) for m in levels]
    ax_b.plot(levels, rr, "o--", color=FLOOR, markersize=3.2, linewidth=1.0, markerfacecolor="white",
              markeredgewidth=0.9, zorder=2)
    ax_b.errorbar(levels, sc, yerr=[sc - lo, hi - sc], fmt="o-", color=INSTRUCTION, ecolor=MUTED,
                  markersize=4, linewidth=1.2, elinewidth=0.8, capsize=2, zorder=3)
    ax_b.plot([1.15, 1.37], [92, 92], "-", color=INSTRUCTION, linewidth=1.2)
    ax_b.plot([1.15, 1.37], [82, 82], "--", color=FLOOR, linewidth=1.0)
    ax_b.text(1.43, 92, "other setups", fontsize=6.8, va="center")
    ax_b.text(1.43, 82, "rerun of default setup", fontsize=6.8, va="center")
    for x, n in zip(levels, ns):
        ax_b.text(x, -7.5, f"n={n}", ha="center", fontsize=6.6, color=MUTED)
    ax_b.set_xlabel("Absolute margin (default setup, run 3)")
    ax_b.set_ylabel("Outcome changes (%)")
    ax_b.set_xticks(levels)
    ax_b.set_ylim(-12, 100)
    ax_b.spines["left"].set_bounds(0, 100)
    ax_b.set_yticks([0, 20, 40, 60, 80, 100])
    ax_b.set_title("Change against decision margin", loc="left")
    for ax in (ax_a, ax_b):
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig2")


def figure_three():
    cells = [DEFAULT] + list(SETUPS)
    X = pd.concat({(c, r): runs(c, [r])[r] for c in cells for r in (["run1", "run2", "run3"] if c == DEFAULT else ["run1", "run2"])},
                  axis=1).dropna()
    base = np.sign(X[(DEFAULT, "run1")])
    counts = np.sort(np.sum([np.sign(X[(c, "run1")]) != base for c in SETUPS], axis=0))
    k = len(SETUPS)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("stab", ["#1f4e5f", "#4a90a4", "#9dc3d0", "#e8b04b", "#c0392b"])
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    cols = 31
    nrows = int(np.ceil(len(counts) / cols))
    for i, v in enumerate(counts):
        r, c = divmod(i, cols)
        ax.add_patch(mpl.patches.Rectangle((c + 0.06, nrows - 1 - r + 0.06), 0.88, 0.88,
                                           facecolor=cmap(v / k), edgecolor="white", linewidth=0.5))
    ax.set_xlim(-0.3, cols + 0.1)
    ax.set_ylim(-0.4, nrows + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, k))
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", pad=0.03, fraction=0.045, aspect=30)
    cb.set_ticks([0, k])
    cb.ax.set_xticklabels(["0", str(k)])
    cb.ax.tick_params(labelsize=7.5, color=MUTED, length=2)
    cb.outline.set_visible(False)
    cb.ax.text(-0.02, 0.5, "stable", transform=cb.ax.transAxes, fontsize=8.5, ha="right", va="center")
    cb.ax.text(1.02, 0.5, "unstable", transform=cb.ax.transAxes, fontsize=8.5, ha="left", va="center")
    save(fig, "fig3")
    print(f"  fig3: {len(counts)} prompts, {int((counts == 0).sum())} unchanged by all {k}, "
          f"{int((counts == k).sum())} changed by all {k}")


def table_pdf(name, header, rows, align, width=5.5, header2=None, group_rules=()):
    """Booktabs-style table drawn as text. header2: optional spanning header [(label, first_col, last_col)]."""
    ncol = len(header)
    lines = len(rows) + 1 + (1 if header2 else 0)
    lh = 0.17
    fig = plt.figure(figsize=(width, lh * (lines + 0.9)))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, lines + 0.9); ax.axis("off")
    # column x positions from the longest entry in each column
    lens = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(header)]
    xs = np.cumsum([0] + lens[:-1]) / sum(lens)
    ws = np.array(lens) / sum(lens)
    def put(y, i, txt, bold=False):
        x = {"l": xs[i] + 0.005, "r": xs[i] + ws[i] - 0.005, "c": xs[i] + ws[i] / 2}[align[i]]
        ax.text(x, y, txt, ha={"l": "left", "r": "right", "c": "center"}[align[i]], va="center", fontsize=7.2,
                fontweight="bold" if bold else "normal")
    top = lines + 0.45
    ax.plot([0, 1], [top, top], color=INK, linewidth=0.8)
    y = top - 0.5
    if header2:
        for label, a, b in header2:
            xa, xb = xs[a], xs[b] + ws[b]
            ax.text((xa + xb) / 2, y, label, ha="center", va="center", fontsize=7.2)
            ax.plot([xa + 0.01, xb - 0.01], [y - 0.42, y - 0.42], color=INK, linewidth=0.4)
        y -= 1
    for i, h in enumerate(header):
        put(y, i, h)
    ax.plot([0, 1], [y - 0.5, y - 0.5], color=INK, linewidth=0.5)
    for k, r in enumerate(rows):
        y -= 1
        if k in group_rules:
            ax.plot([0, 1], [y + 0.5, y + 0.5], color=INK, linewidth=0.4)
        for i, v in enumerate(r):
            put(y, i, str(v))
    ax.plot([0, 1], [y - 0.5, y - 0.5], color=INK, linewidth=0.8)
    save(fig, name)


def ci(v, lo, hi, sign=False, d=1):
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f"{f.format(v)} [{f.format(lo)}, {f.format(hi)}]".replace("-", "\u2212")


def tables(t1, cl):
    t = t1.set_index("cell")
    d = t.loc[DEFAULT]
    main_rows, full_rows = [], []
    for c in ["C1F1D2", "C2F1D1", "C2F1D2", "C1F2D1", "C1F2D2", "C2F2D1", "C2F2D2"]:
        r = t.loc[c]
        common = [r.setup, f"{r.score_run1:.2f}", f"{r.score_run2:.2f}", f"{r.setup_avg_score:.2f}",
                  f"{r.default_avg_score:.2f}", ci(r.score_change, r.score_change_lo, r.score_change_hi, True, 2),
                  ci(r.pct_different_winner, r.pct_different_winner_lo, r.pct_different_winner_hi),
                  f"{r.default_floor:.1f}", f"{r.setup_floor:.1f}"]
        main_rows.append(common + [ci(r.excess_both_floors, r.excess_both_floors_lo, r.excess_both_floors_hi, True)])
        full_rows.append(common + [ci(r.excess_paper, r.excess_paper_lo, r.excess_paper_hi, True),
                                   ci(r.excess_both_floors, r.excess_both_floors_lo, r.excess_both_floors_hi, True)])
    default_row = ["Default setup", f"{d.score_run1:.2f}", f"{d.score_run2:.2f}", f"{d.default_avg_score:.2f}", "",
                   "", "", ci(d.default_floor, d.default_floor_lo, d.default_floor_hi), ""]
    head = ["Setup", "Run 1", "Run 2", "Setup avg", "Default avg", "Score change", "Different winner (%)",
            "Default floor", "Setup floor"]
    table_pdf("table1_main", head + ["Excess, both floors"], [default_row + [""]] + main_rows,
              "lrrrrrrrrr", width=7.0, group_rules=(1,))
    table_pdf("table1_full", head + ["Excess, default floor", "Excess, both floors"],
              [default_row + ["", ""]] + full_rows, "lrrrrrrrrrr", width=7.6, group_rules=(1,))

    cond = pd.read_csv(RES / "step12_conditions_october.csv").set_index("cell")
    rows = []
    for c in ORDER:
        r = cond.loc[c]
        rows.append([LEVELS["C"][c[1]], LEVELS["F"][c[3]], LEVELS["D"][c[5]], int(r.complete_all_runs),
                     int(r.paired_with_default), f"{r.score_run1:.2f}", f"{r.score_run2:.2f}",
                     f"{r.score_run3:.2f}" if pd.notna(r.score_run3) else "", f"{r.tie_rate_avg:.1f}%"])
    table_pdf("table_conditions", ["Criteria", "Output", "Scope", "Complete", "Paired", "Run 1", "Run 2",
                                   "Run 3", "Tie rate"], rows, "lllrrrrrr", width=5.5,
              header2=[("Raw score", 5, 7)])

    th = pd.read_csv(RES / "step13_thresholds_october.csv")
    a, b = th[th.cells.str.startswith("run 1")], th[th.cells == "all 17 runs"]
    rows = [[f"{x.majority_share:.2f}", f"{x.tie_share:.1f}", f"{x.unanimous:.1f}%", f"{x.majority:.1f}%",
             f"{x.unresolved:.1f}%", f"{y.unanimous:.1f}%", f"{y.majority:.1f}%", f"{y.unresolved:.1f}%"]
            for x, y in zip(a.itertuples(), b.itertuples())]
    table_pdf("table_thresholds", ["Majority share", "Tie share", "Unanimous", "Majority", "Unresolved",
                                   "Unanimous", "Majority", "Unresolved"], rows, "rrrrrrrr", width=5.5,
              header2=[("Run 1 of each setup", 2, 4), ("All 17 runs", 5, 7)])

    tal = pd.read_csv(RES / "step12_tally_october.csv")
    rows = [[r.comparison.replace(" -> ", " to "), ci(r.ladder_change, r.ladder_lo, r.ladder_hi, True, 2),
             ci(r.tally_change, r.tally_lo, r.tally_hi, True, 2)] for r in tal.itertuples()]
    table_pdf("table_tally", ["Comparison", "Equal spacing [95% CI]", "Arena-Hard weighted wins [95% CI]"],
              rows, "lrr", width=5.0, group_rules=(3,))

    A = cl[cl.section == "A_by_margin"]
    rows = []
    for m in [0.0, 0.5, 1.0, 1.5, 2.0]:
        g = A[A.group == f"margin {m:g}"].set_index("measure")
        rows.append([f"{m:g}", int(g.loc["default_rerun", "n"]),
                     ci(g.loc["default_rerun", "value"], g.loc["default_rerun", "lo"], g.loc["default_rerun", "hi"]),
                     ci(g.loc["setup_change", "value"], g.loc["setup_change", "lo"], g.loc["setup_change", "hi"]),
                     ci(g.loc["excess_both", "value"], g.loc["excess_both", "lo"], g.loc["excess_both", "hi"], True)])
    table_pdf("table_margin", ["Margin", "Prompts", "Rerun of default setup", "Change vs default setup",
                               "Excess over floor"], rows, "rrrrr", width=5.0)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT, help="Directory for the nine PDFs and PNG previews.")
    OUT = parser.parse_args().output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    t1 = pd.read_csv(RES / "step2_table1_october.csv")
    cl = pd.read_csv(RES / "step10_closeness_october.csv")
    figure_one(t1)
    figure_two(cl)
    figure_three()
    tables(t1, cl)
    print(f"wrote {sorted(p.name for p in OUT.glob('*.pdf'))} to {OUT}")


if __name__ == "__main__":
    main()
