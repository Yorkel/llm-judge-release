"""Across all eight setups: which prompts change, and what kind of change, as parallel tables.

Uses all 17 October runs (default setup C1F1D1: runs 1-3; seven other setups: runs 1-2), on the prompts
with a verdict in every run. The winner of a prompt in a run is the sign of its order-averaged score.

A. Reruns:  how many of the 8 setups give this prompt a different winner when simply run again
            (0 = never, 1-7 = some, 8 = all)
B. Setups:  how many of the 7 other setups give this prompt a different winner from the default setup,
            run 1 against run 1 (0 = never, 1-6 = some, 7 = all)
C. Overall: never changes (same winner in all 17 runs) / changes only between setups (every setup repeats
            its own winner, setups disagree) / changes on a rerun (some setup's runs disagree)
Types of change, as % of all comparisons (unchanged + to or from a tie + reversal = 100):
  reruns  = every pair of runs of the same setup (default 3 pairs + 7 x 1)
  setups  = every pairing of a default run with another setup's run (7 setups x 3 x 2)
  overall = every pair among all 17 runs, same setup or not (136 pairs)
95% intervals: bootstrap over prompts (20,000 resamples, seed 20260830). Reads data only.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from repeat_floor import load

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "october_rerun"
OUT = HERE / "results" / "october_rerun" / "step5_breakdowns_october.csv"
DEFAULT = "C1F1D1"
OTHERS = ["C1F1D2", "C2F1D1", "C2F1D2", "C1F2D1", "C1F2D2", "C2F2D1", "C2F2D2"]
SETUPS = [DEFAULT] + OTHERS
BOOTSTRAP = 20000
SEED = 20260830


def load_all():
    cols = {(DEFAULT, r): load(DATA / r / f"{DEFAULT}.jsonl") for r in ("run1", "run2", "run3")}
    cols.update({(c, r): load(DATA / r / f"{c}.jsonl") for c in OTHERS for r in ("run1", "run2")})
    return np.sign(pd.concat(cols, axis=1).dropna())


def counts(W, cols):
    idx = {s: [k for k, (c, _) in enumerate(cols) if c == s] for s in SETUPS}
    n_rerun = np.stack([(W[..., idx[s]] != W[..., idx[s][:1]]).any(axis=-1) for s in SETUPS], axis=-1).sum(axis=-1)
    n_setup = np.stack([W[..., idx[s][0]] != W[..., idx[DEFAULT][0]] for s in OTHERS], axis=-1).sum(axis=-1)
    same_all = (W == W[..., :1]).all(axis=-1)
    return idx, n_rerun, n_setup, same_all


def measures(W, cols):
    idx, n_rerun, n_setup, same_all = counts(W, cols)
    out = {
        "A_never": 100 * (n_rerun == 0).mean(axis=-1),
        "A_some": 100 * ((n_rerun > 0) & (n_rerun < 8)).mean(axis=-1),
        "A_all": 100 * (n_rerun == 8).mean(axis=-1),
        "B_never": 100 * (n_setup == 0).mean(axis=-1),
        "B_some": 100 * ((n_setup > 0) & (n_setup < 7)).mean(axis=-1),
        "B_all": 100 * (n_setup == 7).mean(axis=-1),
        "C_never_changes": 100 * same_all.mean(axis=-1),
        "C_only_between_setups": 100 * (~same_all & (n_rerun == 0)).mean(axis=-1),
        "C_on_a_rerun": 100 * (n_rerun > 0).mean(axis=-1),
    }
    rerun_pairs = [(i, j) for s in SETUPS for i, j in combinations(idx[s], 2)]
    setup_pairs = [(i, j) for s in OTHERS for i, j in product(idx[DEFAULT], idx[s])]
    overall_pairs = list(combinations(range(len(cols)), 2))  # every pair among all 17 runs
    for label, pairs in (("rerun", rerun_pairs), ("setup", setup_pairs), ("overall", overall_pairs)):
        a = np.stack([W[..., i] for i, _ in pairs], axis=-1)
        b = np.stack([W[..., j] for _, j in pairs], axis=-1)
        reversal = (a * b) == -1
        tie = (a != b) & ~reversal
        out[f"{label}_unchanged"] = 100 * (a == b).mean(axis=(-2, -1))
        out[f"{label}_to_or_from_tie"] = 100 * tie.mean(axis=(-2, -1))
        out[f"{label}_reversal"] = 100 * reversal.mean(axis=(-2, -1))
    return out


def main():
    W_df = load_all()
    cols = list(W_df.columns)
    W = W_df.to_numpy()
    rng = np.random.default_rng(SEED)
    point = measures(W, cols)
    idx = rng.integers(0, len(W), size=(BOOTSTRAP, len(W)))
    chunks = [measures(W[part], cols) for part in np.array_split(idx, max(1, BOOTSTRAP // 1000))]  # memory
    boot = {k: np.concatenate([c[k] for c in chunks]) for k in point}
    rows = []
    for k, v in point.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append({"measure": k, "value": float(v), "lo": float(lo), "hi": float(hi)})
        print(f"{k:28s} {v:6.1f}  [{lo:5.1f}, {hi:5.1f}]")
    _, n_rerun, n_setup, _ = counts(W, cols)
    for name, n, top in (("A reruns (of 8 setups)", n_rerun, 8), ("B setups (of 7 setups)", n_setup, 7)):
        dist = np.bincount(n, minlength=top + 1)
        print(f"{name}: " + ", ".join(f"{k}: {c} ({100 * c / len(W):.1f}%)" for k, c in enumerate(dist)))
        rows += [{"measure": f"{name[0]}_count_{k}", "value": int(c), "lo": np.nan, "hi": np.nan} for k, c in enumerate(dist)]
    pd.DataFrame(rows).assign(n_prompts=len(W)).to_csv(OUT, index=False)
    print(f"\nprompts: {len(W)}\nwrote {OUT}")


if __name__ == "__main__":
    main()
