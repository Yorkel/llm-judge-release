"""October threshold-based classification (paper.tex:398-430, Table thresholds).

Each prompt is unanimous (every cell gives the same sign class), majority (one sign class holds at
least the majority share and the tie share is not above the tie threshold) or unresolved, using the
paper's rule unchanged (_canonical_defs.assign_bucket: missing cells dropped, all 500 prompts).
Two versions of the cells: run 1 of each of the eight GPT-4.1 setups (as in May, one run per setup),
and all 17 October runs. Sweep: majority share 0.50 / 0.60 / 0.75 x tie share 0.3 / 0.4 / 0.5.
Reads data only.
"""
from itertools import product
from pathlib import Path

import pandas as pd

from _canonical_defs import assign_bucket
from october_step2_table1 import DEFAULT, SETUPS, runs

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "results" / "october_rerun" / "step13_thresholds_october.csv"
LABELS = {"B1_stable": "unanimous", "B2_sensitive": "majority", "B3_unresolved": "unresolved"}


def main():
    cells = [DEFAULT] + list(SETUPS)
    run_names = {c: ["run1", "run2", "run3"] if c == DEFAULT else ["run1", "run2"] for c in cells}
    versions = {
        "run 1 of each setup (8 cells)": pd.concat({c: runs(c, ["run1"])["run1"] for c in cells}, axis=1),
        "all 17 runs": pd.concat({f"{c}_{r}": runs(c, [r])[r] for c in cells for r in run_names[c]}, axis=1),
    }
    rows = []
    for version, wide in versions.items():
        for maj, tie in product([0.5, 0.6, 0.75], [0.3, 0.4, 0.5]):
            b = wide.apply(assign_bucket, axis=1, majority_thresh=maj, tie_thresh=tie).map(LABELS)
            share = b.value_counts(normalize=True) * 100
            rows.append({"cells": version, "majority_share": maj, "tie_share": tie, "n_prompts": len(b),
                         **{k: share.get(k, 0.0) for k in ("unanimous", "majority", "unresolved")}})
    table = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.round(1).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
