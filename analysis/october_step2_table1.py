"""October Table 1: each GPT-4.1 setup against the default setup (C1F1D1).

Default setup: October runs 1-3. Other setups: October runs 1-2.
For each setup, on the prompts with a verdict in all five runs involved (3 default + 2 setup):
  - scores (0-100, equal spacing): setup run 1, run 2, setup average, default average, change
  - % of prompts with a different winner from the default setup, averaged over the 6 pairings of runs
  - default floor: % with a different winner between default runs, averaged over its 3 pairs
  - setup floor: % with a different winner between the setup's two runs
  - excess, paper's way: change - default floor
  - excess, both floors: change - (default floor + setup floor) / 2
95% intervals: paired bootstrap over prompts (20,000 resamples, seed 20260830), with all runs of a
prompt drawn together. Reads data only; makes no API calls.
"""
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from repeat_floor import load  # order-averaged score per prompt; NaN if either answer order failed

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "october_rerun"
OUT = HERE / "results" / "october_rerun" / "step2_table1_october.csv"
DEFAULT = "C1F1D1"
SETUPS = {
    "C1F1D2": "+ description",
    "C2F1D1": "Accuracy-first",
    "C2F1D2": "Accuracy + description",
    "C1F2D1": "Label only",
    "C1F2D2": "Label only + description",
    "C2F2D1": "Accuracy + label only",
    "C2F2D2": "Accuracy + label + description",
}
BOOTSTRAP = 20000
SEED = 20260830


def runs(cell, names):
    return pd.concat({n: load(DATA / n / f"{cell}.jsonl") for n in names}, axis=1)


def pct_different(a, b):
    """% of prompts whose winner (sign of the score) differs between two runs."""
    return 100 * np.mean(np.sign(a) != np.sign(b), axis=-1)


def default_floor(D):
    """Average over the default setup's 3 pairs of runs (1-2, 1-3, 2-3)."""
    return np.mean([pct_different(D[..., i], D[..., j]) for i, j in combinations(range(3), 2)], axis=0)


def measures(D, S):
    """D: (..., prompts, 3) default scores; S: (..., prompts, 2) setup scores."""
    setup_runs = 50 + 25 * S.mean(axis=-2)
    setup_avg = setup_runs.mean(axis=-1)
    default_avg = 50 + 25 * D.mean(axis=(-2, -1))
    change = np.mean([pct_different(D[..., i], S[..., j]) for i, j in product(range(3), range(2))], axis=0)
    d_floor = default_floor(D)
    s_floor = pct_different(S[..., 0], S[..., 1])
    return {
        "score_run1": setup_runs[..., 0],
        "score_run2": setup_runs[..., 1],
        "setup_avg_score": setup_avg,
        "default_avg_score": default_avg,
        "score_change": setup_avg - default_avg,
        "pct_different_winner": change,
        "default_floor": d_floor,
        "setup_floor": s_floor,
        "excess_paper": change - d_floor,
        "excess_both_floors": change - (d_floor + s_floor) / 2,
    }


def with_intervals(point, boot):
    row = {}
    for k, v in point.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        row[k], row[f"{k}_lo"], row[f"{k}_hi"] = float(v), float(lo), float(hi)
    return row


def main():
    rng = np.random.default_rng(SEED)
    d = runs(DEFAULT, ["run1", "run2", "run3"])
    rows = []

    # Default setup row: its own three runs, on prompts complete in all three
    D = d.dropna().to_numpy()
    idx = rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))
    point = {**{f"score_run{k + 1}": 50 + 25 * D[:, k].mean() for k in range(3)},
             "default_avg_score": 50 + 25 * D.mean(),
             **{f"floor_{i + 1}{j + 1}": pct_different(D[:, i], D[:, j]) for i, j in combinations(range(3), 2)},
             "default_floor": default_floor(D)}
    Db = D[idx]
    boot = {**{f"score_run{k + 1}": 50 + 25 * Db[..., k].mean(axis=-1) for k in range(3)},
            "default_avg_score": 50 + 25 * Db.mean(axis=(-2, -1)),
            **{f"floor_{i + 1}{j + 1}": pct_different(Db[..., i], Db[..., j]) for i, j in combinations(range(3), 2)},
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
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print(table[[c for c in table.columns if not c.endswith(("_lo", "_hi"))]].round(2).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
