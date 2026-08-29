"""Canonical definitions for paper-facing analyses.

Single source of truth for:
- Cell IDs and axis mapping
- Verdict → candidate-relative score mapping (round-aware sign flip)
- Disagreement filters (any-tier, boundary-flip, strict-reversal,
  same-side magnitude shift, strong-to-weak shift)
- Bucket rule (B1/B2/B3 with 60% majority + 40% tie thresholds)

Imported by both `analysis/build_canonical.py` (script) and the
exploratory notebooks (`10_*.ipynb`, `11_*.ipynb`).

If you change a definition here, every paper-facing number changes.
Don't fork; if a notebook needs an exploratory variant, copy the function
locally with a clear name suffix (e.g., `boundary_flip_strict`) so the
canonical name keeps a single binding.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---- Cell / axis identifiers --------------------------------------------------

ANCHOR = "C1F1D1"

L2_CELLS = [
    "C1F1D1",
    "C1F1D2",
    "C1F2D1",
    "C1F2D2",
    "C2F1D1",
    "C2F1D2",
    "C2F2D1",
    "C2F2D2",
]

AXIS_CELLS = {
    "C": "C2F1D1",
    "F": "C1F2D1",
    "D": "C1F1D2",
    "L1_claude": "L1_claude",
    "L1_gemini": "L1_gemini",
}


# ---- Verdict → score ----------------------------------------------------------

VERDICT_SCORE = {"A>>B": -2, "A>B": -1, "A=B": 0, "B>A": 1, "B>>A": 2}


def candidate_score(verdict: str | None, round_name: str) -> float:
    """Candidate-relative ordinal score in [-2, +2].

    Round 1 has A=baseline, B=candidate (positive score = candidate wins).
    Round 2 has positions swapped, so sign is inverted.
    """
    if verdict is None or verdict not in VERDICT_SCORE:
        return np.nan
    base = VERDICT_SCORE[verdict]
    return float(base if round_name == "round1" else -base)


# ---- Sign + disagreement filters ----------------------------------------------

def sign(score: float) -> int:
    """Sign of a (possibly NaN) candidate score. 0 for ties and NaN."""
    if pd.isna(score) or score == 0:
        return 0
    return 1 if score > 0 else -1


def any_tier_disagreement(anchor_score: float, variant_score: float) -> bool:
    """Any difference in ordinal tier (broadest filter)."""
    return pd.notna(anchor_score) and pd.notna(variant_score) and anchor_score != variant_score


def boundary_flip(anchor_score: float, variant_score: float) -> bool:
    """Sign-flip relative to anchor, treating ties as their own sign-class."""
    return (
        pd.notna(anchor_score)
        and pd.notna(variant_score)
        and sign(anchor_score) != sign(variant_score)
    )


def strict_reversal(anchor_score: float, variant_score: float) -> bool:
    """Candidate-favoured to baseline-favoured (or vice versa). Excludes ties."""
    return (
        pd.notna(anchor_score)
        and pd.notna(variant_score)
        and sign(anchor_score) * sign(variant_score) == -1
    )


def same_side_magnitude_shift(anchor_score: float, variant_score: float) -> bool:
    """Same sign, different magnitude. e.g. A>>B → A>B (does not cross boundary)."""
    if not (pd.notna(anchor_score) and pd.notna(variant_score)):
        return False
    return sign(anchor_score) == sign(variant_score) != 0 and abs(anchor_score) != abs(variant_score)


def strong_to_weak_shift(anchor_score: float, variant_score: float) -> bool:
    """Decisive (|score|=2) ↔ narrow (|score|=1) on the same side."""
    if not same_side_magnitude_shift(anchor_score, variant_score):
        return False
    return {abs(anchor_score), abs(variant_score)} == {1.0, 2.0}


# ---- Bucket rule --------------------------------------------------------------

def assign_bucket(
    row: pd.Series,
    majority_thresh: float = 0.6,
    tie_thresh: float = 0.4,
) -> str:
    """B1 stable / B2 sensitive / B3 unresolved.

    A prompt is B1 if one verdict tier wins ≥`majority_thresh` of cells and the
    tie share is below `tie_thresh`. B3 if no tier reaches majority OR the tie
    share is at or above `tie_thresh`. B2 otherwise.

    Drops NaN cells from the calculation. If fewer than half the cells are
    present after the drop, returns B3_unresolved (matches legacy notebooks).
    """
    signs = np.sign(row.dropna())
    if len(signs) < max(2, len(row) // 2):
        return "B3_unresolved"
    pos = int((signs > 0).sum())
    neg = int((signs < 0).sum())
    zero = int((signs == 0).sum())
    n = len(signs)
    if pos == n or neg == n or zero == n:
        return "B1_stable"
    majority = max(pos, neg, zero) / n
    if majority < majority_thresh or zero > n * tie_thresh:
        return "B3_unresolved"
    return "B2_sensitive"
