"""October direct reversals: each GPT-4.1 setup against the default setup (C1F1D1).

A direct reversal is a prompt whose winner switches from one model to the other, not via a tie:
sign(score in one run) x sign(score in the other) = -1 (paper.tex:379). Tied prompts stay in the
denominator. Default setup: October runs 1-3. Other setups: October runs 1-2.
For each setup, on the prompts with a verdict in all five runs involved (3 default + 2 setup):
  - % of prompts reversed against the default setup, averaged over the 6 pairings of runs
  - default reversal floor: % reversed between default runs, averaged over its 3 pairs
  - setup reversal floor: % reversed between the setup's two runs
  - excess, paper's way: reversals - default floor
  - excess, both floors: reversals - (default floor + setup floor) / 2
95% intervals: paired bootstrap over prompts (20,000 resamples, seed 20260830), with all runs of a
prompt drawn together. Reads data only; makes no API calls.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from october_step2_table1 import BOOTSTRAP, DEFAULT, SEED, SETUPS, runs, with_intervals

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "october_rerun" / "step8_reversals_october.csv"


def pct_reversed(a, b):
    """% of prompts whose winner switches directly from one model to the other between two runs."""
    return 100 * np.mean(np.sign(a) * np.sign(b) == -1, axis=-1)


def default_floor(D):
    """Average over the default setup's 3 pairs of runs (1-2, 1-3, 2-3)."""
    return np.mean([pct_reversed(D[..., i], D[..., j]) for i, j in combinations(range(3), 2)], axis=0)


def measures(D, S):
    """D: (..., prompts, 3) default scores; S: (..., prompts, 2) setup scores."""
    reversed_ = np.mean([pct_reversed(D[..., i], S[..., j]) for i, j in product(range(3), range(2))], axis=0)
    d_floor = default_floor(D)
    s_floor = pct_reversed(S[..., 0], S[..., 1])
    return {
        "pct_reversed": reversed_,
        "default_floor": d_floor,
        "setup_floor": s_floor,
        "excess_paper": reversed_ - d_floor,
        "excess_both_floors": reversed_ - (d_floor + s_floor) / 2,
    }


def main():
    rng = np.random.default_rng(SEED)
    d = runs(DEFAULT, ["run1", "run2", "run3"])
    rows = []

    # Default setup row: its own three runs, on prompts complete in all three
    D = d.dropna().to_numpy()
    Db = D[rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))]
    pairs = list(combinations(range(3), 2))
    point = {**{f"floor_{i + 1}{j + 1}": pct_reversed(D[:, i], D[:, j]) for i, j in pairs},
             "default_floor": default_floor(D)}
    boot = {**{f"floor_{i + 1}{j + 1}": pct_reversed(Db[..., i], Db[..., j]) for i, j in pairs},
            "default_floor": default_floor(Db)}
    rows.append({"setup": "Default setup", "cell": DEFAULT, "n_prompts": len(D), **with_intervals(point, boot)})

    # One row per setup, on prompts complete in all five runs involved
    for cell, name in SETUPS.items():
        both = pd.concat({"d": d, "s": runs(cell, ["run1", "run2"])}, axis=1).dropna()
        D, S = both["d"].to_numpy(), both["s"].to_numpy()
        idx = rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))
        rows.append({"setup": name, "cell": cell, "n_prompts": len(D),
                     **with_intervals(measures(D, S), measures(D[idx], S[idx]))})

    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    for r in rows:
        if r["cell"] == DEFAULT:
            print(f"{r['setup']:31s} n={r['n_prompts']}  reversal floor {r['default_floor']:.2f} "
                  f"[{r['default_floor_lo']:.2f}, {r['default_floor_hi']:.2f}]  pairs "
                  + ", ".join(f"{p}: {r[f'floor_{p}']:.2f}" for p in ("12", "13", "23")))
            continue
        print(f"{r['setup']:31s} n={r['n_prompts']}  reversed {r['pct_reversed']:5.2f} | default floor "
              f"{r['default_floor']:.2f} | setup floor {r['setup_floor']:.2f} | excess paper "
              f"{r['excess_paper']:+.2f} [{r['excess_paper_lo']:+.2f}, {r['excess_paper_hi']:+.2f}] | "
              f"both floors {r['excess_both_floors']:+.2f} [{r['excess_both_floors_lo']:+.2f}, "
              f"{r['excess_both_floors_hi']:+.2f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
