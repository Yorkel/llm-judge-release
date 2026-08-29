"""Compare the unchanged-condition repeat against the original baseline run.

Both runs use the identical specification: Arena-Hard criteria, reasoning
required, no scope statement, GPT-4.1, temperature zero, both answer orders.
Any disagreement between them is run-to-run variation, not a specification
effect. This is the floor the between-condition rates in the paper should be
read against.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _canonical_defs import candidate_score

HERE = Path(__file__).resolve().parent.parent
ORIGINAL = HERE / "data" / "full_factorial" / "C1F1D1.jsonl"
REPEAT = HERE / "data" / "repeat_run" / "C1F1D1.jsonl"
BOOTSTRAP = 2000
SEED = 20260830


def load(path: Path) -> pd.Series:
    """Order-averaged candidate score per uid; NaN if either round failed."""
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            s1 = candidate_score(r["round1"].get("verdict"), "round1")
            s2 = candidate_score(r["round2"].get("verdict"), "round2")
            rows[r["uid"]] = np.nan if (pd.isna(s1) or pd.isna(s2)) else (s1 + s2) / 2
    return pd.Series(rows, name=path.parent.name)


def boot_ci(flags: np.ndarray, rng) -> tuple[float, float]:
    n = len(flags)
    draws = rng.integers(0, n, size=(BOOTSTRAP, n))
    b = 100.0 * flags[draws].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    rng = np.random.default_rng(SEED)
    a, b = load(ORIGINAL), load(REPEAT)
    df = pd.concat([a.rename("run1"), b.rename("run2")], axis=1).dropna()
    n = len(df)
    print(f"prompts complete in both runs: {n}")

    boundary = (np.sign(df.run1) != np.sign(df.run2)).to_numpy()
    reversal = (np.sign(df.run1) * np.sign(df.run2) == -1).to_numpy()
    tier = (df.run1 != df.run2).to_numpy()

    for label, flags in [("boundary change", boundary),
                         ("direct reversal", reversal),
                         ("any tier change", tier)]:
        lo, hi = boot_ci(flags, rng)
        print(f"  {label:16s} {100*flags.mean():5.1f}%  [{lo:.1f}, {hi:.1f}]  ({flags.sum()}/{n})")

    d = 25.0 * (df.run2 - df.run1)
    se = d.std(ddof=1) / np.sqrt(n)
    print(f"\npaired aggregate change run2 - run1: {d.mean():+.2f} points "
          f"[{d.mean()-1.96*se:+.2f}, {d.mean()+1.96*se:+.2f}]")
    print(f"raw score run1 {25*df.run1.mean()+50:.2f}   run2 {25*df.run2.mean()+50:.2f}")

    # Does repeat disagreement concentrate near the boundary, as specification
    # change does? Reported for the same margin bins as Figure 2.
    print("\nrepeat boundary change by |baseline margin|:")
    for m in [0.0, 0.5, 1.0, 1.5, 2.0]:
        sel = boundary[np.isclose(df.run1.abs(), m)]
        if len(sel):
            print(f"  margin {m:.1f}: {100*sel.mean():5.1f}%  (n={len(sel)})")

    out = HERE / "results" / "canonical_v1" / "repeat_floor.csv"
    pd.DataFrame([{
        "n_paired": n,
        "boundary_change_pct": 100 * boundary.mean(),
        "direct_reversal_pct": 100 * reversal.mean(),
        "any_tier_pct": 100 * tier.mean(),
        "paired_aggregate_change_pp": d.mean(),
    }]).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
