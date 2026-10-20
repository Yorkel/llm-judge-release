"""Reviewer 2EEv's third question: are the prompts that flip on a rerun the same ones that change with the setup?

Run flip:   the winner differs between two runs of the default setup.
Setup flip: the winner differs between the THIRD default run and a run of the setup.
The two flips never share a default run, so one odd run cannot create both by construction.
All six assignments are used (which default run is the third x which setup run) and averaged.

Per setup: % run flip, % setup flip, % both, % both expected if unrelated (run x setup),
ratio observed / expected (as in paper.tex:204), and the share of setup-changed prompts that also
flip on a rerun. 95% intervals: bootstrap over prompts (20,000 resamples, seed 20260830). Reads data only.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from repeat_floor import load

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "october_rerun"
OUT = HERE / "results" / "october_rerun" / "step4_overlap_october.csv"
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
# (rerun pair from the default setup, the remaining default run) for all three choices of "third" run
DEFAULT_SPLITS = [((0, 1), 2), ((0, 2), 1), ((1, 2), 0)]
BOOTSTRAP = 20000
SEED = 20260830


def measures(D, S):
    """D: (..., prompts, 3) default winners; S: (..., prompts, 2) setup winners (signs)."""
    r_list, s_list, both_list, exp_list = [], [], [], []
    for ((i, j), k), m in product(DEFAULT_SPLITS, range(2)):
        run = D[..., i] != D[..., j]
        setup = D[..., k] != S[..., m]
        r, s = run.mean(axis=-1), setup.mean(axis=-1)
        r_list.append(r)
        s_list.append(s)
        both_list.append((run & setup).mean(axis=-1))
        exp_list.append(r * s)
    r, s, both, expected = (np.mean(x, axis=0) for x in (r_list, s_list, both_list, exp_list))
    return {
        "pct_run_flip": 100 * r,
        "pct_setup_flip": 100 * s,
        "pct_both": 100 * both,
        "pct_both_if_unrelated": 100 * expected,
        "ratio_observed_to_expected": both / expected,
        "share_of_setup_changes_that_also_flip_on_rerun": 100 * both / s,
    }


def main():
    rng = np.random.default_rng(SEED)
    d = pd.concat({r: load(DATA / r / f"{DEFAULT}.jsonl") for r in ("run1", "run2", "run3")}, axis=1)
    rows = []
    for cell, name in SETUPS.items():
        s = pd.concat({r: load(DATA / r / f"{cell}.jsonl") for r in ("run1", "run2")}, axis=1)
        both = pd.concat({"d": d, "s": s}, axis=1).dropna()
        D, S = np.sign(both["d"].to_numpy()), np.sign(both["s"].to_numpy())
        point = measures(D, S)
        idx = rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))
        boot = measures(D[idx], S[idx])
        row = {"setup": name, "cell": cell, "n_prompts": len(D)}
        for k, v in point.items():
            lo, hi = np.percentile(boot[k], [2.5, 97.5])
            row[k], row[f"{k}_lo"], row[f"{k}_hi"] = float(v), float(lo), float(hi)
        rows.append(row)
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    for _, r in table.iterrows():
        print(f"{r.setup:31s} n={r.n_prompts} | rerun flip {r.pct_run_flip:.1f}% | setup flip {r.pct_setup_flip:.1f}% | "
              f"both {r.pct_both:.1f}% (if unrelated {r.pct_both_if_unrelated:.1f}%) | "
              f"ratio {r.ratio_observed_to_expected:.2f} [{r.ratio_observed_to_expected_lo:.2f}, {r.ratio_observed_to_expected_hi:.2f}] | "
              f"share {r.share_of_setup_changes_that_also_flip_on_rerun:.1f}% "
              f"[{r.share_of_setup_changes_that_also_flip_on_rerun_lo:.1f}, {r.share_of_setup_changes_that_also_flip_on_rerun_hi:.1f}]")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
