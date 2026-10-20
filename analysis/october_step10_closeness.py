"""October closeness: do close calls change more? (paper.tex:116-120, Figure 2, Figure 3)

The margin of a prompt is |score| in the default setup's run 3 only. Every change is counted from
default runs 1-2 and the setups' runs 1-2 only, so margin and change come from independent runs.
Prompts: those with a verdict in all 17 October runs.

A. By margin (0, 0.5, 1, 1.5, 2), the % of prompts whose winner changes:
   - default rerun: default run 1 vs run 2
   - setups' own reruns: each setup's run 1 vs run 2, averaged over the 7 setups
   - floor (both floors): the average of those two
   - setup change: default runs 1-2 vs setup runs 1-2 (4 pairings), averaged over the 7 setups
   - excess: setup change - floor (both floors); and setup change - default rerun (May's way)
B. Mean margin of changed vs unchanged prompts (default run 1 vs run 1), for accuracy-first (as at
   paper.tex:116) and for the default rerun, with two-sided Mann-Whitney tests.
C. How many of the 7 setups change each prompt against the default setup (run 1 vs run 1; Figure 2
   left and Figure 3), and the mean margin of prompts changed by none of them vs by all of them.
95% intervals: bootstrap over prompts within each group (20,000 resamples, seed 20260830). Reads data only.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from october_step2_table1 import BOOTSTRAP, DEFAULT, SEED, SETUPS, runs

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "october_rerun" / "step10_closeness_october.csv"
MARGINS = [0.0, 0.5, 1.0, 1.5, 2.0]


def load():
    cols = {(DEFAULT, r): runs(DEFAULT, [r])[r] for r in ("run1", "run2", "run3")}
    for cell in SETUPS:
        cols.update({(cell, r): runs(cell, [r])[r] for r in ("run1", "run2")})
    return pd.concat(cols, axis=1).dropna()


def ci(x, rng):
    x = np.asarray(x, float)
    boot = x[rng.integers(0, len(x), size=(BOOTSTRAP, len(x)))].mean(axis=1)
    return x.mean(), *np.percentile(boot, [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    X = load()
    W = np.sign(X)
    margin = X[(DEFAULT, "run3")].abs()
    d1, d2 = W[(DEFAULT, "run1")], W[(DEFAULT, "run2")]

    per = pd.DataFrame(index=X.index)
    per["default_rerun"] = 100.0 * (d1 != d2)
    per["setups_own_rerun"] = 100 * np.mean([W[(c, "run1")] != W[(c, "run2")] for c in SETUPS], axis=0)
    per["floor_both"] = (per.default_rerun + per.setups_own_rerun) / 2
    per["setup_change"] = 100 * np.mean([W[(DEFAULT, a)] != W[(c, b)] for c in SETUPS
                                         for a in ("run1", "run2") for b in ("run1", "run2")], axis=0)
    per["excess_both"] = per.setup_change - per.floor_both
    per["excess_vs_default_rerun"] = per.setup_change - per.default_rerun

    rows = []
    for m in MARGINS:
        part = per[margin == m]
        for k in per.columns:
            v, lo, hi = ci(part[k], rng)
            rows.append({"section": "A_by_margin", "group": f"margin {m:g}", "measure": k, "n": len(part),
                         "value": v, "lo": lo, "hi": hi})

    for label, changed in (("accuracy_first", W[(DEFAULT, "run1")] != W[("C2F1D1", "run1")]),
                           ("default_rerun", d1 != d2)):
        a, b = margin[changed], margin[~changed]
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue
        for name, g in (("changed", a), ("unchanged", b)):
            v, lo, hi = ci(g, rng)
            rows.append({"section": "B_margin_changed_vs_unchanged", "group": f"{label}: {name}",
                         "measure": "mean_margin", "n": len(g), "value": v, "lo": lo, "hi": hi,
                         "mann_whitney_p": p})

    n_changed = np.sum([W[(DEFAULT, "run1")] != W[(c, "run1")] for c in SETUPS], axis=0)
    for k in range(len(SETUPS) + 1):
        rows.append({"section": "C_setups_changing_prompt", "group": f"{k} of 7", "measure": "prompts",
                     "n": int((n_changed == k).sum()), "value": 100 * (n_changed == k).mean()})
    never, always = margin[n_changed == 0], margin[n_changed == len(SETUPS)]
    p = mannwhitneyu(never, always, alternative="two-sided").pvalue
    for name, g in (("never changes (0 of 7)", never), ("always changes (7 of 7)", always)):
        v, lo, hi = ci(g, rng)
        rows.append({"section": "C_margin_never_vs_always", "group": name, "measure": "mean_margin",
                     "n": len(g), "value": v, "lo": lo, "hi": hi, "mann_whitney_p": p})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)

    print(f"prompts: {len(X)}\n\nA. % changing, by margin in default run 3")
    A = table[table.section == "A_by_margin"]
    for m in MARGINS:
        g = A[A.group == f"margin {m:g}"].set_index("measure")
        f = lambda k: f"{g.loc[k, 'value']:5.1f} [{g.loc[k, 'lo']:5.1f}, {g.loc[k, 'hi']:5.1f}]"
        print(f"  margin {m:g} (n={int(g.n.iloc[0]):3d}) | default rerun {f('default_rerun')} | setups' own "
              f"{g.loc['setups_own_rerun', 'value']:5.1f} | floor both {g.loc['floor_both', 'value']:5.1f} | "
              f"setup change {f('setup_change')} | excess both {f('excess_both')} | vs default rerun "
              f"{g.loc['excess_vs_default_rerun', 'value']:+5.1f}")
    print("\nB. mean margin (run 3), changed vs unchanged (run 1 vs run 1)")
    for _, r in table[table.section == "B_margin_changed_vs_unchanged"].iterrows():
        print(f"  {r.group:28s} n={r.n:3d}  {r.value:.2f} [{r.lo:.2f}, {r.hi:.2f}]  Mann-Whitney p={r.mann_whitney_p:.2g}")
    print("\nC. how many of the 7 setups change each prompt (vs default, run 1 vs run 1)")
    print("  " + ", ".join(f"{r.group}: {r.n} ({r.value:.1f}%)" for _, r in
                           table[table.section == "C_setups_changing_prompt"].iterrows()))
    for _, r in table[table.section == "C_margin_never_vs_always"].iterrows():
        print(f"  {r.group:24s} n={r.n:3d}  mean margin {r.value:.2f} [{r.lo:.2f}, {r.hi:.2f}]  "
              f"Mann-Whitney p={r.mann_whitney_p:.2g}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
