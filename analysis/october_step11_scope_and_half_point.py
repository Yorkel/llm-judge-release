"""Two October checks for Appendix C and paper.tex:132.

A. Coding-only excess (paper.tex:55, 211): Table 1's coding-description row (C1F1D2 against the default
   setup) on the coding prompts alone, the maths prompts alone, and all prompts. Same measures as
   october_step2_table1.py: % with a different winner (6 pairings), default floor, setup floor,
   excess (paper's way) and excess (both floors), with paired bootstrap intervals.
B. The +-0.5 share (paper.tex:132), May's way (results/exploratory_checks.md section 2): of the changes
   between the default setup and the seven other GPT-4.1 setups, the % where either score is +-0.5,
   against the base rate (the % of all compared pairs where either score is +-0.5). Pooled over all
   pairings of runs, and run 1 against run 1 as in May.
95% intervals: bootstrap over prompts (20,000 resamples, seed 20260830). Reads data only.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from october_step2_table1 import BOOTSTRAP, DEFAULT, SEED, SETUPS, measures, runs, with_intervals

HERE = Path(__file__).resolve().parent.parent
PROMPTS = HERE / "data" / "preprocessed_pairs.jsonl"
OUT = HERE / "results" / "october_rerun" / "step11_scope_and_half_point_october.csv"
KEYS = ["pct_different_winner", "default_floor", "setup_floor", "excess_paper", "excess_both_floors"]


def main():
    rng = np.random.default_rng(SEED)
    subject = pd.Series({json.loads(l)["uid"]: json.loads(l)["subcategory"] for l in PROMPTS.open()})
    d = runs(DEFAULT, ["run1", "run2", "run3"])
    rows = []

    # A. Coding-description row by subject
    both = pd.concat({"d": d, "s": runs("C1F1D2", ["run1", "run2"])}, axis=1).dropna()
    for label, keep in (("all", slice(None)), ("coding", subject.reindex(both.index) == "coding"),
                        ("maths", subject.reindex(both.index) == "math")):
        part = both[keep]
        D, S = part["d"].to_numpy(), part["s"].to_numpy()
        idx = rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))
        m, mb = measures(D, S), measures(D[idx], S[idx])
        rows.append({"section": "A_coding_description_by_subject", "group": label, "n": len(D),
                     **with_intervals({k: m[k] for k in KEYS}, {k: mb[k] for k in KEYS})})

    # B. +-0.5 share among setup changes, pooled over the seven setups
    for version, pairings in (("all pairings", list(product(range(3), range(2)))), ("run 1 vs run 1", [(0, 0)])):
        changed_half = changed = pairs_half = pairs = 0
        for cell in SETUPS:
            b = pd.concat({"d": d, "s": runs(cell, ["run1", "run2"])}, axis=1).dropna()
            D, S = b["d"].to_numpy(), b["s"].to_numpy()
            for i, j in pairings:
                ch = np.sign(D[:, i]) != np.sign(S[:, j])
                half = (np.abs(D[:, i]) == 0.5) | (np.abs(S[:, j]) == 0.5)
                changed_half += (ch & half).sum(); changed += ch.sum()
                pairs_half += half.sum(); pairs += len(ch)
        share, base = 100 * changed_half / changed, 100 * pairs_half / pairs
        rows.append({"section": "B_half_point_share", "group": version, "n": int(changed),
                     "share_of_changes_with_half": share, "base_rate": base, "elevation": share / base})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print("A. coding description vs default setup, by subject")
    for r in rows[:3]:
        print(f"  {r['group']:6s} n={r['n']:3d} | change {r['pct_different_winner']:5.1f} | default floor "
              f"{r['default_floor']:5.1f} | setup floor {r['setup_floor']:5.1f} | excess paper "
              f"{r['excess_paper']:+5.1f} [{r['excess_paper_lo']:+5.1f}, {r['excess_paper_hi']:+5.1f}] | excess both "
              f"{r['excess_both_floors']:+5.1f} [{r['excess_both_floors_lo']:+5.1f}, {r['excess_both_floors_hi']:+5.1f}]")
    print("B. +-0.5 share among changes between the default setup and the seven setups")
    for r in rows[3:]:
        print(f"  {r['group']:15s} changes={r['n']:5d} | involving +-0.5 {r['share_of_changes_with_half']:.1f}% "
              f"vs base rate {r['base_rate']:.1f}% | elevation {r['elevation']:.2f}x")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
