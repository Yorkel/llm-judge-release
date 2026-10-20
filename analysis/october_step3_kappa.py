"""October self-agreement: linearly weighted Cohen's kappa between runs of the same setup.

Scores are re-binned to five tiers exactly as in the paper (Appendix H; 11_inter_judge_agreement.ipynb):
cut points -1.5, -0.5, 0.5, 1.5, central tier closed at both ends, so a score of +-0.5 counts as a tie.
Default setup: kappa for each of its 3 pairs of runs and their average. Other setups: run 1 vs run 2.
95% intervals: bootstrap over prompts (20,000 resamples, seed 20260830). Reads data only.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from repeat_floor import load

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "october_rerun"
OUT = HERE / "results" / "october_rerun" / "step3_kappa_october.csv"
CELLS = {
    "C1F1D1": "Default setup",
    "C1F1D2": "+ description",
    "C2F1D1": "Accuracy-first",
    "C2F1D2": "Accuracy + description",
    "C1F2D1": "Label only",
    "C1F2D2": "Label only + description",
    "C2F2D1": "Accuracy + label only",
    "C2F2D2": "Accuracy + label + description",
}
TIERS = [-2, -1, 0, 1, 2]
BOOTSTRAP = 20000
SEED = 20260830


def to_tier(s):
    return np.select([s < -1.5, s < -0.5, s <= 0.5, s <= 1.5], [-2, -1, 0, 1], default=2)


def kappa(a, b):
    return cohen_kappa_score(a, b, labels=TIERS, weights="linear")


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for cell, name in CELLS.items():
        runs = ["run1", "run2", "run3"] if cell == "C1F1D1" else ["run1", "run2"]
        T = to_tier(pd.concat({r: load(DATA / r / f"{cell}.jsonl") for r in runs}, axis=1).dropna().to_numpy())
        pairs = list(combinations(range(len(runs)), 2))
        point = [kappa(T[:, i], T[:, j]) for i, j in pairs]
        idx = rng.integers(0, len(T), size=(BOOTSTRAP, len(T)))
        boot = np.array([[kappa(T[ix, i], T[ix, j]) for i, j in pairs] for ix in idx])
        avg, avg_boot = np.mean(point), boot.mean(axis=1)
        lo, hi = np.percentile(avg_boot, [2.5, 97.5])
        row = {"setup": name, "cell": cell, "n_prompts": len(T), "kappa": avg, "kappa_lo": lo, "kappa_hi": hi}
        if len(pairs) > 1:
            for (i, j), k, kb in zip(pairs, point, boot.T):
                plo, phi = np.percentile(kb, [2.5, 97.5])
                row[f"kappa_{i + 1}{j + 1}"], row[f"kappa_{i + 1}{j + 1}_lo"], row[f"kappa_{i + 1}{j + 1}_hi"] = k, plo, phi
        rows.append(row)
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    for _, r in table.iterrows():
        extra = ""
        if pd.notna(r.get("kappa_12", np.nan)):
            extra = " | pairs " + ", ".join(f"{p}: {r[f'kappa_{p}']:.3f}" for p in ("12", "13", "23"))
        print(f"{r.setup:31s} n={r.n_prompts}  kappa {r.kappa:.3f} [{r.kappa_lo:.3f}, {r.kappa_hi:.3f}]{extra}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
