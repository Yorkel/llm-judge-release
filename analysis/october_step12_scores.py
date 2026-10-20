"""October aggregate scores: conditions table, score range, factorial averages, Arena-Hard tally check.

A. Conditions table (paper.tex:321) and score range (paper.tex:79): for each GPT-4.1 setup and run, the
   prompts complete (both answer orders parsed), the prompts paired with the default setup (complete
   in all five runs compared: default 1-3, setup 1-2), the raw score on its own complete prompts
   (0-100, equal spacing) and the tie rate (order-averaged score exactly zero).
B. Factorial averages (paper.tex:112, 197-204): for each choice (criteria, output, scope), the paired
   score change and the % of prompts with a different winner for each of the four pairs of setups
   that differ only in that choice, and their average. Prompts complete in all 17 runs; each setup's
   score averaged over its runs; change rates averaged over all pairings of runs.
C. Arena-Hard tally check (paper.tex:65, Appendix D): each setup's paired score change from the default
   setup under equal spacing and under Arena-Hard's weighted wins (show_result.py, weight 3: "much
   better" = 3 wins, "slightly better" = 1, tie = half a win each; each answer order counted
   separately; no style control). Also the default setup's own rerun changes under both rules.
95% intervals: bootstrap over prompts (seed 20260830). Reads data only.
"""
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from _canonical_defs import candidate_score
from october_step2_table1 import BOOTSTRAP, DEFAULT, SEED, SETUPS

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "october_rerun"
OUT = HERE / "results" / "october_rerun"
CELLS = [DEFAULT] + list(SETUPS)
NAMES = {DEFAULT: "Default setup", **SETUPS}
RUNS = {c: ["run1", "run2", "run3"] if c == DEFAULT else ["run1", "run2"] for c in CELLS}
FACTORS = {"criteria (Arena-Hard -> accuracy-first)": 1, "output (reasoning -> label only)": 3,
           "scope (none -> coding description)": 5}  # character position of the level in "C1F1D1"


def rounds(cell, run):
    """Per-round candidate scores (-2..2) per uid: DataFrame with columns r1, r2."""
    rows = {}
    for line in (DATA / run / f"{cell}.jsonl").open():
        r = json.loads(line)
        rows[r["uid"]] = (candidate_score(r["round1"].get("verdict"), "round1"),
                          candidate_score(r["round2"].get("verdict"), "round2"))
    return pd.DataFrame.from_dict(rows, orient="index", columns=["r1", "r2"]).astype(float)


def ladder(R):
    """Equal-spacing score 0-100 over every verdict in R (prompts x [runs x] answer orders)."""
    return 50 + 25 * R.mean()


def tally(R):
    """Arena-Hard weighted-win share 0-100 for the candidate, over every verdict in R."""
    games = np.where(np.abs(R) == 2, 3.0, 1.0)
    wins = np.where(R > 0, games, 0.0) + np.where(R == 0, 0.5, 0.0)
    return 100 * wins.sum() / games.sum()


def boot(f, arrays, rng, n):
    idx = rng.integers(0, n, size=(BOOTSTRAP, n))
    vals = np.array([f(*[a[i] for a in arrays]) for i in idx])
    return np.percentile(vals, [2.5, 97.5])


def main():
    rng = np.random.default_rng(SEED)
    R = {(c, r): rounds(c, r) for c in CELLS for r in RUNS[c]}

    # A. Conditions table
    complete = {k: v.dropna() for k, v in R.items()}
    rows = []
    for c in CELLS:
        own = pd.concat([complete[(c, r)] for r in RUNS[c]], axis=1, join="inner").index
        paired = own.intersection(pd.concat([complete[(DEFAULT, r)] for r in RUNS[DEFAULT]], axis=1, join="inner").index)
        row = {"cell": c, "setup": NAMES[c], "complete_all_runs": len(own), "paired_with_default": len(paired)}
        for r in RUNS[c]:
            v = complete[(c, r)]
            s = v.mean(axis=1)
            row[f"complete_{r}"], row[f"score_{r}"], row[f"tie_rate_{r}"] = len(v), 50 + 25 * s.mean(), 100 * (s == 0).mean()
        row["score_avg"] = np.mean([row[f"score_{r}"] for r in RUNS[c]])
        row["tie_rate_avg"] = np.mean([row[f"tie_rate_{r}"] for r in RUNS[c]])
        rows.append(row)
    cond = pd.DataFrame(rows)
    cond.to_csv(OUT / "step12_conditions_october.csv", index=False)

    # B. Factorial averages, on prompts complete in all 17 runs
    uids = pd.concat([complete[k] for k in R], axis=1, join="inner").index
    A = {k: R[k].loc[uids].to_numpy() for k in R}
    fac_rows = []
    for factor, pos in FACTORS.items():
        pairs = [(a, a[:pos] + "2" + a[pos + 1:]) for a in CELLS if a[pos] == "1"]

        def stats(*arrs):
            out = []
            for k, (a, b) in enumerate(pairs):
                Ra, Rb = arrs[2 * k], arrs[2 * k + 1]  # (n, runs, 2)
                change = ladder(Rb) - ladder(Ra)
                sa, sb = np.sign(Ra.mean(axis=-1)), np.sign(Rb.mean(axis=-1))
                rate = 100 * np.mean([sa[:, i] != sb[:, j] for i in range(sa.shape[1]) for j in range(sb.shape[1])])
                out += [change, rate]
            return np.array(out)

        arrays = []
        for a, b in pairs:
            arrays += [np.stack([A[(a, r)] for r in RUNS[a]], axis=1), np.stack([A[(b, r)] for r in RUNS[b]], axis=1)]
        point = stats(*arrays)
        idx = rng.integers(0, len(uids), size=(BOOTSTRAP, len(uids)))
        bt = np.array([stats(*[x[i] for x in arrays]) for i in idx])
        for k, (a, b) in enumerate(pairs + [("average", "average")]):
            if a == "average":
                pc, pr = point[0::2].mean(), point[1::2].mean()
                bc, br = bt[:, 0::2].mean(axis=1), bt[:, 1::2].mean(axis=1)
            else:
                pc, pr, bc, br = point[2 * k], point[2 * k + 1], bt[:, 2 * k], bt[:, 2 * k + 1]
            fac_rows.append({"factor": factor, "from": a, "to": b, "n_prompts": len(uids),
                             "score_change": pc, "score_change_lo": np.percentile(bc, 2.5), "score_change_hi": np.percentile(bc, 97.5),
                             "pct_different_winner": pr, "pct_lo": np.percentile(br, 2.5), "pct_hi": np.percentile(br, 97.5)})
    fac = pd.DataFrame(fac_rows)
    fac.to_csv(OUT / "step12_factorial_october.csv", index=False)

    # C. Arena-Hard tally check: paired changes from the default setup under both rules
    tal_rows = []
    D_all = pd.concat({r: complete[(DEFAULT, r)] for r in RUNS[DEFAULT]}, axis=1, join="inner")
    D = np.stack([D_all[r].to_numpy() for r in RUNS[DEFAULT]], axis=1)
    for i, j in combinations(range(3), 2):  # the default setup's own reruns
        f = lambda X, i=i, j=j: np.array([ladder(X[:, j]) - ladder(X[:, i]), tally(X[:, j]) - tally(X[:, i])])
        p = f(D)
        b = np.array([f(D[ix]) for ix in rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))])
        tal_rows.append({"comparison": f"Default setup, run {i + 1} -> run {j + 1}", "n_prompts": len(D),
                         "ladder_change": p[0], "ladder_lo": np.percentile(b[:, 0], 2.5), "ladder_hi": np.percentile(b[:, 0], 97.5),
                         "tally_change": p[1], "tally_lo": np.percentile(b[:, 1], 2.5), "tally_hi": np.percentile(b[:, 1], 97.5)})
    for c in SETUPS:
        paired = D_all.index
        for r in RUNS[c]:
            paired = paired.intersection(complete[(c, r)].index)
        Dp = np.stack([complete[(DEFAULT, r)].loc[paired].to_numpy() for r in RUNS[DEFAULT]], axis=1)
        Sp = np.stack([complete[(c, r)].loc[paired].to_numpy() for r in RUNS[c]], axis=1)
        f = lambda X, Y: np.array([ladder(Y) - ladder(X), tally(Y) - tally(X)])
        p = f(Dp, Sp)
        idx = rng.integers(0, len(Dp), size=(BOOTSTRAP, len(Dp)))
        b = np.array([f(Dp[ix], Sp[ix]) for ix in idx])
        tal_rows.append({"comparison": NAMES[c], "n_prompts": len(Dp),
                         "ladder_change": p[0], "ladder_lo": np.percentile(b[:, 0], 2.5), "ladder_hi": np.percentile(b[:, 0], 97.5),
                         "tally_change": p[1], "tally_lo": np.percentile(b[:, 1], 2.5), "tally_hi": np.percentile(b[:, 1], 97.5)})
    tal = pd.DataFrame(tal_rows)
    tal.to_csv(OUT / "step12_tally_october.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("A. CONDITIONS\n", cond.round(2).to_string(index=False))
    print(f"\nscore range across the 8 setups (average of runs): {cond.score_avg.min():.2f} to {cond.score_avg.max():.2f}; "
          f"across all 17 runs: {min(cond.filter(like='score_run').min()):.2f} to {max(cond.filter(like='score_run').max()):.2f}")
    print("\nB. FACTORIAL\n", fac.round(2).to_string(index=False))
    print("\nC. TALLY\n", tal.round(2).to_string(index=False))
    print(f"\nwrote {OUT}/step12_conditions_october.csv, step12_factorial_october.csv, step12_tally_october.csv")


if __name__ == "__main__":
    main()
