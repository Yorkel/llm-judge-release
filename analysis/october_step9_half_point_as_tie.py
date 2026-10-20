"""Reviewer 2EEv's second question: what if a score of +-0.5 counted as a tie? (October; appendix check)

A score of +-0.5 can combine a tie and a slight preference, or opposing slight and strong
preferences across the two answer orders. The paper's rule
takes the sign, so it counts as a win (paper.tex:132). This check recounts every prompt's winner with
|score| <= 0.5 as a tie, in all 17 October runs, and redoes Table 1's prompt-level columns under both
rules, side by side: % with a different winner from the default setup (6 pairings), default floor
(3 pairs), setup floor, excess (paper's way) and excess (both floors). Aggregate scores are unaffected
and not repeated. Also: the share of scores at +-0.5 per setup, and, as in paper.tex:132, how often
the default setup's rerun changes involve a +-0.5 score against the base rate.
95% intervals: paired bootstrap over prompts (20,000 resamples, seed 20260830). Reads data only.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from october_step2_table1 import (BOOTSTRAP, DEFAULT, SEED, SETUPS, default_floor, measures, pct_different,
                                  runs, with_intervals)

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "october_rerun" / "step9_half_point_as_tie_october.csv"
KEYS = ["pct_different_winner", "default_floor", "setup_floor", "excess_paper", "excess_both_floors"]


def half_as_tie(x):
    return np.where(np.abs(x) <= 0.5, 0.0, x)


def both_rules(D, S):
    cur, new = measures(D, S), measures(half_as_tie(D), half_as_tie(S))
    out = {}
    for k in KEYS:
        out[f"{k}_current"], out[f"{k}_half_tie"], out[f"{k}_shift"] = cur[k], new[k], new[k] - cur[k]
    return out


def main():
    rng = np.random.default_rng(SEED)
    d = runs(DEFAULT, ["run1", "run2", "run3"])
    rows = []

    # Default setup: its floor under both rules, and how much of its rerun change involves +-0.5
    D = d.dropna().to_numpy()
    pairs = list(combinations(range(3), 2))
    changed = np.concatenate([np.sign(D[:, i]) != np.sign(D[:, j]) for i, j in pairs])
    half = np.concatenate([(np.abs(D[:, i]) == 0.5) | (np.abs(D[:, j]) == 0.5) for i, j in pairs])
    Db = D[rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))]
    point = {"default_floor_current": default_floor(D), "default_floor_half_tie": default_floor(half_as_tie(D))}
    boot = {"default_floor_current": default_floor(Db), "default_floor_half_tie": default_floor(half_as_tie(Db))}
    point["default_floor_shift"] = point["default_floor_half_tie"] - point["default_floor_current"]
    boot["default_floor_shift"] = boot["default_floor_half_tie"] - boot["default_floor_current"]
    rows.append({"setup": "Default setup", "cell": DEFAULT, "n_prompts": len(D),
                 "pct_scores_at_half": 100 * np.mean(np.abs(D) == 0.5),
                 "pct_rerun_changes_involving_half": 100 * half[changed].mean(),
                 "pct_rerun_pairs_involving_half": 100 * half.mean(),
                 **with_intervals(point, boot)})

    for cell, name in SETUPS.items():
        both = pd.concat({"d": d, "s": runs(cell, ["run1", "run2"])}, axis=1).dropna()
        D, S = both["d"].to_numpy(), both["s"].to_numpy()
        idx = rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))
        rows.append({"setup": name, "cell": cell, "n_prompts": len(D),
                     "pct_scores_at_half": 100 * np.mean(np.abs(S) == 0.5),
                     **with_intervals(both_rules(D, S), both_rules(D[idx], S[idx]))})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    r = rows[0]
    print(f"Default setup n={r['n_prompts']}  scores at +-0.5: {r['pct_scores_at_half']:.1f}%  | floor "
          f"{r['default_floor_current']:.1f} -> {r['default_floor_half_tie']:.1f} (shift {r['default_floor_shift']:+.1f} "
          f"[{r['default_floor_shift_lo']:+.1f}, {r['default_floor_shift_hi']:+.1f}])  | rerun changes involving "
          f"+-0.5: {r['pct_rerun_changes_involving_half']:.1f}% vs {r['pct_rerun_pairs_involving_half']:.1f}% of all pairs")
    for r in rows[1:]:
        print(f"{r['setup']:31s} at +-0.5 {r['pct_scores_at_half']:4.1f}% | change "
              f"{r['pct_different_winner_current']:5.1f} -> {r['pct_different_winner_half_tie']:5.1f} | setup floor "
              f"{r['setup_floor_current']:4.1f} -> {r['setup_floor_half_tie']:4.1f} | default floor "
              f"{r['default_floor_current']:4.1f} -> {r['default_floor_half_tie']:4.1f} | excess both "
              f"{r['excess_both_floors_current']:5.1f} -> {r['excess_both_floors_half_tie']:5.1f} "
              f"[{r['excess_both_floors_half_tie_lo']:4.1f}, {r['excess_both_floors_half_tie_hi']:4.1f}] | excess paper "
              f"{r['excess_paper_current']:5.1f} -> {r['excess_paper_half_tie']:5.1f} "
              f"[{r['excess_paper_half_tie_lo']:4.1f}, {r['excess_paper_half_tie_hi']:4.1f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
