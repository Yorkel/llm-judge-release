"""October agreement between setups: linearly weighted Cohen's kappa, each setup against the default setup.

Same five-tier re-binning as the paper and october_step3_kappa.py (+-0.5 counts as a tie).
Each setup's kappa with the default setup is averaged over all 6 pairings of runs (3 default x 2 setup),
shown beside each setup's kappa with itself (from step 3) and the default setup's kappa with itself.
The self-agreement references use their own complete samples, whose sizes are reported separately
as setup_self_n_prompts and default_self_n_prompts; n_prompts is the between-setup matched sample.
95% intervals: bootstrap over prompts (20,000 resamples, seed 20260830). Reads data only.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from october_step3_kappa import kappa, to_tier
from repeat_floor import load

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "october_rerun"
OUT = HERE / "results" / "october_rerun" / "step3b_kappa_between_october.csv"
SELF = HERE / "results" / "october_rerun" / "step3_kappa_october.csv"
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


def main():
    rng = np.random.default_rng(SEED)
    self_k = pd.read_csv(SELF).set_index("cell")
    d = pd.concat({r: load(DATA / r / f"{DEFAULT}.jsonl") for r in ("run1", "run2", "run3")}, axis=1)
    rows = []
    for cell, name in SETUPS.items():
        s = pd.concat({r: load(DATA / r / f"{cell}.jsonl") for r in ("run1", "run2")}, axis=1)
        both = pd.concat({"d": d, "s": s}, axis=1).dropna()
        D, S = to_tier(both["d"].to_numpy()), to_tier(both["s"].to_numpy())
        pairings = list(product(range(3), range(2)))
        point = np.mean([kappa(D[:, i], S[:, j]) for i, j in pairings])
        idx = rng.integers(0, len(D), size=(BOOTSTRAP, len(D)))
        boot = np.array([np.mean([kappa(D[ix, i], S[ix, j]) for i, j in pairings]) for ix in idx])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append({"setup": name, "cell": cell, "n_prompts": len(D),
                     "kappa_with_default": point, "kappa_with_default_lo": lo, "kappa_with_default_hi": hi,
                     "setup_kappa_with_itself": self_k.loc[cell, "kappa"],
                     "default_kappa_with_itself": self_k.loc[DEFAULT, "kappa"],
                     "setup_self_n_prompts": int(self_k.loc[cell, "n_prompts"]),
                     "default_self_n_prompts": int(self_k.loc[DEFAULT, "n_prompts"])})
        print(f"{name:31s} n={len(D)}  with default {point:.3f} [{lo:.3f}, {hi:.3f}]  | "
              f"itself {self_k.loc[cell, 'kappa']:.3f} | default itself {self_k.loc[DEFAULT, 'kappa']:.3f}")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
