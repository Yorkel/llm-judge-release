"""Build canonical paper-facing analysis outputs for the EMNLP draft.

This script reads frozen raw judge JSONL files and prompt metadata, defines the
core disagreement filters once, and writes versioned outputs under
``results/canonical_v1``. Existing top-level ``results/*.csv`` files are treated
as legacy exploratory outputs.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


from _canonical_defs import (
    ANCHOR,
    L2_CELLS,
    AXIS_CELLS,
    VERDICT_SCORE,
    candidate_score,
    sign,
    any_tier_disagreement,
    boundary_flip,
    strict_reversal,
    same_side_magnitude_shift,
    strong_to_weak_shift,
    assign_bucket,
)
from repeat_floor import ORIGINAL as REPEAT_ORIGINAL
from repeat_floor import REPEAT as REPEAT_PATH
from repeat_floor import load as load_repeat_scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "canonical_v1"
OUT.mkdir(parents=True, exist_ok=True)

CELL_PATHS = {cell: ROOT / "data" / "full_factorial" / f"{cell}.jsonl" for cell in L2_CELLS}
CELL_PATHS["L1_claude"] = ROOT / "data" / "L1_swap" / f"{ANCHOR}.jsonl"
CELL_PATHS["L1_gemini"] = ROOT / "data" / "L1_swap_v2_flash_16k" / f"{ANCHOR}.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    for cell, path in CELL_PATHS.items():
        if not path.exists():
            continue
        for rec in load_jsonl(path):
            s1 = candidate_score(rec.get("round1", {}).get("verdict"), "round1")
            s2 = candidate_score(rec.get("round2", {}).get("verdict"), "round2")
            mean_score = np.nan if pd.isna(s1) or pd.isna(s2) else (s1 + s2) / 2
            long_rows.append(
                {
                    "uid": rec["uid"],
                    "cell_id": cell,
                    "r1_score": s1,
                    "r2_score": s2,
                    "mean_score": mean_score,
                    "r1_parse_success": bool(rec.get("round1", {}).get("parse_success")),
                    "r2_parse_success": bool(rec.get("round2", {}).get("parse_success")),
                    "r1_output_tokens": rec.get("round1", {}).get("output_tokens", np.nan),
                    "r2_output_tokens": rec.get("round2", {}).get("output_tokens", np.nan),
                    "model": rec.get("model"),
                }
            )
    long = pd.DataFrame(long_rows)
    wide = long.pivot(index="uid", columns="cell_id", values="mean_score")
    return long, wide


CODE_BLOCK_RE = re.compile(r"```.*?```", flags=re.DOTALL)
MD_RE = re.compile(r"(`[^`]+`|#{1,6}\s|\*\*|__|\[[^\]]+\]\([^)]+\)|^\s*[-*+]\s)", flags=re.MULTILINE)


def compute_style_features(meta: pd.DataFrame) -> pd.DataFrame:
    exact_path = ROOT / "results" / "style_features.csv"
    if exact_path.exists():
        # The original notebook used Arena-Hard's markdown feature extractor.
        # The extractor itself is not present in this workspace, so the exact
        # generated feature table is imported and versioned into canonical_v1.
        return pd.read_csv(exact_path)

    rows = []
    for _, row in meta.iterrows():
        cand = str(row["candidate_response"])
        base = str(row["baseline_response"])
        c_len = len(cand.split())
        b_len = len(base.split())
        c_clean = CODE_BLOCK_RE.sub("", cand)
        b_clean = CODE_BLOCK_RE.sub("", base)
        c_md = len(MD_RE.findall(c_clean))
        b_md = len(MD_RE.findall(b_clean))
        length_diff = (c_len - b_len) / (c_len + b_len + 1e-9)
        md_diff = (c_md - b_md) / (c_md + b_md + 1e-9)
        rows.append({"uid": row["uid"], "length_diff": length_diff, "md_diff": md_diff})
    style = pd.DataFrame(rows).set_index("uid")
    for col in ["length_diff", "md_diff"]:
        std = style[col].std(ddof=0)
        style[f"{col}_z"] = 0.0 if std == 0 else (style[col] - style[col].mean()) / std
    return style.reset_index()


def style_control_tables(wide: pd.DataFrame, style: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    style_i = style.set_index("uid")
    rows = []
    coef = {}
    for cell in wide.columns:
        df = wide[[cell]].join(style_i[["length_diff", "md_diff", "length_diff_z", "md_diff_z"]], how="inner").dropna()
        if df.empty:
            continue
        y = (df[cell] + 2) / 4
        reg_raw = LinearRegression().fit(df[["length_diff", "md_diff"]].values, y.values)
        reg_z = LinearRegression().fit(df[["length_diff_z", "md_diff_z"]].values, y.values)
        coef[cell] = (float(reg_raw.coef_[0]), float(reg_raw.coef_[1]))
        rows.append(
            {
                "cell_id": cell,
                "n": len(df),
                "raw_winequiv_pct": float(y.mean() * 100),
                "adjusted_winequiv_pct": float(reg_raw.intercept_ * 100),
                "delta_pp": float((y.mean() - reg_raw.intercept_) * 100),
                "length_coef_raw": float(reg_raw.coef_[0]),
                "md_coef_raw": float(reg_raw.coef_[1]),
                "length_coef_z": float(reg_z.coef_[0]),
                "md_coef_z": float(reg_z.coef_[1]),
            }
        )
    control = pd.DataFrame(rows)

    anchor_len, anchor_md = coef[ANCHOR]
    fixed_rows = []
    anchor_fixed = None
    for cell in wide.columns:
        df = wide[[cell]].join(style_i[["length_diff", "md_diff"]], how="inner").dropna()
        if df.empty:
            continue
        y = (df[cell] + 2) / 4
        fixed = float((y - anchor_len * df["length_diff"] - anchor_md * df["md_diff"]).mean() * 100)
        if cell == ANCHOR:
            anchor_fixed = fixed
        fixed_rows.append({"cell_id": cell, "n": len(df), "fixed_anchor_coef_winequiv_pct": fixed})
    fixed = pd.DataFrame(fixed_rows)
    fixed["delta_vs_anchor_fixed_pp"] = fixed["fixed_anchor_coef_winequiv_pct"] - anchor_fixed
    fixed = fixed.merge(
        control[["cell_id", "adjusted_winequiv_pct"]],
        on="cell_id",
        how="left",
    )
    anchor_adj = float(control.loc[control.cell_id == ANCHOR, "adjusted_winequiv_pct"].iloc[0])
    fixed["per_cell_delta_vs_anchor_pp"] = fixed["adjusted_winequiv_pct"] - anchor_adj
    return control, fixed


def disagreement_bridge(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in [c for c in wide.columns if c != ANCHOR]:
        df = wide[[ANCHOR, cell]].dropna()
        a = df[ANCHOR]
        b = df[cell]
        any_mask = [any_tier_disagreement(x, y) for x, y in zip(a, b)]
        boundary_mask = [boundary_flip(x, y) for x, y in zip(a, b)]
        strict_mask = [strict_reversal(x, y) for x, y in zip(a, b)]
        mag_mask = [same_side_magnitude_shift(x, y) for x, y in zip(a, b)]
        strong_weak_mask = [strong_to_weak_shift(x, y) for x, y in zip(a, b)]
        n = len(df)
        rows.append(
            {
                "cell_id": cell,
                "n": n,
                "any_tier_n": int(np.sum(any_mask)),
                "any_tier_pct": 100 * float(np.mean(any_mask)),
                "boundary_flip_n": int(np.sum(boundary_mask)),
                "boundary_flip_pct": 100 * float(np.mean(boundary_mask)),
                "strict_reversal_n": int(np.sum(strict_mask)),
                "strict_reversal_pct": 100 * float(np.mean(strict_mask)),
                "same_side_magnitude_shift_n": int(np.sum(mag_mask)),
                "same_side_magnitude_shift_pct": 100 * float(np.mean(mag_mask)),
                "strong_weak_shift_n": int(np.sum(strong_weak_mask)),
                "strong_weak_shift_pct": 100 * float(np.mean(strong_weak_mask)),
            }
        )
    return pd.DataFrame(rows)


def d_axis_coding_only(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["uid", ANCHOR, "C1F1D2"]
    merged = wide[cols].merge(meta[["uid", "subcategory", "language"]], on="uid", how="left").dropna(subset=[ANCHOR, "C1F1D2"])
    rows = []
    for label, df in [("all", merged), ("coding", merged[merged.subcategory == "coding"]), ("math", merged[merged.subcategory == "math"])]:
        a = df[ANCHOR]
        b = df["C1F1D2"]
        rows.append(
            {
                "subset": label,
                "n": len(df),
                "any_tier_n": sum(any_tier_disagreement(x, y) for x, y in zip(a, b)),
                "any_tier_pct": 100 * np.mean([any_tier_disagreement(x, y) for x, y in zip(a, b)]),
                "boundary_flip_n": sum(boundary_flip(x, y) for x, y in zip(a, b)),
                "boundary_flip_pct": 100 * np.mean([boundary_flip(x, y) for x, y in zip(a, b)]),
                "strict_reversal_n": sum(strict_reversal(x, y) for x, y in zip(a, b)),
                "strict_reversal_pct": 100 * np.mean([strict_reversal(x, y) for x, y in zip(a, b)]),
            }
        )
    return pd.DataFrame(rows)


def language_by_subcategory(wide: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["uid", ANCHOR] + list(AXIS_CELLS.values())
    df = wide[cols].merge(meta[["uid", "subcategory", "language"]], on="uid", how="left")
    df["lang_group"] = np.where(df["language"] == "English", "English", "non-English")
    rows = []
    for axis, cell in AXIS_CELLS.items():
        sub = df.dropna(subset=[ANCHOR, cell]).copy()
        sub["boundary_flip"] = [boundary_flip(x, y) for x, y in zip(sub[ANCHOR], sub[cell])]
        for (subcategory, lang_group), g in sub.groupby(["subcategory", "lang_group"]):
            rows.append(
                {
                    "axis": axis,
                    "cell_id": cell,
                    "subcategory": subcategory,
                    "lang_group": lang_group,
                    "n": len(g),
                    "boundary_flip_n": int(g["boundary_flip"].sum()),
                    "boundary_flip_pct": 100 * float(g["boundary_flip"].mean()),
                }
            )
    return pd.DataFrame(rows)


def bucket_summary(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    subsets = {
        "all_8_prompt_layer": L2_CELLS,
        "F1_only_4cells": ["C1F1D1", "C1F1D2", "C2F1D1", "C2F1D2"],
        "framing_excluded_F1_plus_claude": ["C1F1D1", "C1F1D2", "C2F1D1", "C2F1D2", "L1_claude"],
        "F2_only_4cells": ["C1F2D1", "C1F2D2", "C2F2D1", "C2F2D2"],
    }
    summary_rows = []
    assignment_frames = []
    for subset, cells in subsets.items():
        avail = [cell for cell in cells if cell in wide.columns]
        buckets = wide[avail].apply(assign_bucket, axis=1).rename("bucket")
        counts = buckets.value_counts()
        for bucket in ["B1_stable", "B2_sensitive", "B3_unresolved"]:
            summary_rows.append(
                {
                    "subset": subset,
                    "cells": ";".join(avail),
                    "bucket": bucket,
                    "n_prompts": len(buckets),
                    "bucket_n": int(counts.get(bucket, 0)),
                    "bucket_pct": 100 * float(counts.get(bucket, 0) / len(buckets)),
                    "majority_thresh": 0.6,
                    "tie_thresh": 0.4,
                    "min_valid_cells_rule": "at least max(2, floor(k/2)) non-missing cells",
                }
            )
        assignment_frames.append(
            pd.DataFrame({"uid": wide.index, "subset": subset, "bucket": buckets.values})
        )
    return pd.DataFrame(summary_rows), pd.concat(assignment_frames, ignore_index=True)


def magnitude_shift_breakdown(bridge: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "cell_id",
        "n",
        "same_side_magnitude_shift_n",
        "same_side_magnitude_shift_pct",
        "strong_weak_shift_n",
        "strong_weak_shift_pct",
        "strict_reversal_n",
        "strict_reversal_pct",
    ]
    return bridge[cols].copy()


def prompt_feature_frame(meta: pd.DataFrame) -> pd.DataFrame:
    code_re = re.compile(r"```")
    num_re = re.compile(r"\d")
    out = meta[["uid", "subcategory", "language", "prompt"]].copy()
    out["prompt_len_words"] = out["prompt"].astype(str).str.split().map(len)
    out["prompt_has_code"] = out["prompt"].astype(str).map(lambda x: bool(code_re.search(x)))
    out["prompt_has_numbers"] = out["prompt"].astype(str).map(lambda x: bool(num_re.search(x)))
    out["is_non_english"] = out["language"] != "English"
    return out


def axis_fingerprint_tests(wide: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = prompt_feature_frame(meta).set_index("uid")
    rows = []
    log_rows = []
    masks = {}
    for axis, cell in AXIS_CELLS.items():
        df = wide[[ANCHOR, cell]].join(feats, how="inner").dropna(subset=[ANCHOR, cell])
        df["boundary_flip"] = [boundary_flip(x, y) for x, y in zip(df[ANCHOR], df[cell])]
        masks[axis] = df["boundary_flip"]
        for feature in ["prompt_len_words", "prompt_has_code", "prompt_has_numbers", "is_non_english"]:
            f = df[df.boundary_flip][feature].astype(float)
            n = df[~df.boundary_flip][feature].astype(float)
            try:
                _, p = mannwhitneyu(f, n, alternative="two-sided", method="asymptotic")
            except Exception:
                p = np.nan
            rows.append(
                {
                    "axis": axis,
                    "feature": feature,
                    "boundary_flip_n": int(df.boundary_flip.sum()),
                    "flipped_mean": float(f.mean()),
                    "nonflipped_mean": float(n.mean()),
                    "mean_diff": float(f.mean() - n.mean()),
                    "mw_pvalue": p,
                }
            )

        X = df[["prompt_len_words", "prompt_has_code", "prompt_has_numbers"]].astype(float)
        y = df["boundary_flip"].astype(int)
        if y.nunique() > 1:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, penalty=None))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="'penalty' was deprecated in version 1.8",
                    category=FutureWarning,
                )
                model.fit(X, y)
            coefs = model.named_steps["logisticregression"].coef_[0]
            for name, coef in zip(X.columns, coefs):
                log_rows.append({"axis": axis, "predictor": name, "standardized_logit_coef": float(coef), "n": len(df)})
    tests = pd.DataFrame(rows)
    tests["bh_qvalue"] = bh_qvalues(tests["mw_pvalue"].values)
    tests["bonferroni_p"] = np.minimum(tests["mw_pvalue"] * tests["mw_pvalue"].notna().sum(), 1.0)
    return tests, pd.DataFrame(log_rows)


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    if len(pv) == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order]
    q = ranked * len(ranked) / (np.arange(len(ranked)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    out[valid] = np.minimum(q[inv], 1.0)
    return out


def claude_output_lengths(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell in ["L1_claude"]:
        sub = long[long.cell_id == cell]
        vals = pd.concat([sub["r1_output_tokens"], sub["r2_output_tokens"]]).dropna()
        cap = 4096
        rows.append(
            {
                "cell_id": cell,
                "n_rounds": len(vals),
                "mean_output_tokens": float(vals.mean()),
                "p50_output_tokens": float(vals.quantile(0.50)),
                "p90_output_tokens": float(vals.quantile(0.90)),
                "p95_output_tokens": float(vals.quantile(0.95)),
                "p99_output_tokens": float(vals.quantile(0.99)),
                "max_output_tokens": float(vals.max()),
                "within_10pct_cap_n": int((vals >= 0.9 * cap).sum()),
                "within_10pct_cap_pct": 100 * float((vals >= 0.9 * cap).mean()),
                "at_or_above_cap_n": int((vals >= cap).sum()),
                "token_cap": cap,
            }
        )
    return pd.DataFrame(rows)


def repeat_floor_table() -> pd.DataFrame:
    """Run-to-run variation under the unchanged baseline specification."""
    run1 = load_repeat_scores(REPEAT_ORIGINAL)
    run2 = load_repeat_scores(REPEAT_PATH)
    paired = pd.concat(
        [run1.rename("run1"), run2.rename("run2")],
        axis=1,
    ).dropna()
    boundary = np.sign(paired.run1) != np.sign(paired.run2)
    reversal = np.sign(paired.run1) * np.sign(paired.run2) == -1
    tier = paired.run1 != paired.run2
    aggregate_change = 25.0 * (paired.run2 - paired.run1)
    return pd.DataFrame(
        [
            {
                "n_paired": len(paired),
                "boundary_change_pct": 100 * boundary.mean(),
                "direct_reversal_pct": 100 * reversal.mean(),
                "any_tier_pct": 100 * tier.mean(),
                "paired_aggregate_change_pp": aggregate_change.mean(),
            }
        ]
    )


def write_manifest() -> pd.DataFrame:
    rows = [
        {
            "paper_section": "Abstract/Results 4.2",
            "claim_id": "bucket_full_design",
            "claim_text_short": "44.4% unresolved across all eight prompt-layer cells",
            "csv_file": "bucket_summary.csv",
            "row_filter": "subset=all_8_prompt_layer; bucket=B3_unresolved",
            "value_column": "bucket_pct",
            "filter_definition": "B3 if no sign reaches 60% majority or tie share exceeds 40%; at least max(2, floor(k/2)) valid cells required",
            "source_raw_files": "data/full_factorial/*.jsonl",
            "generated_by": "analysis/build_canonical.py",
            "notes": "Centralizes legacy bucket rule from 10_analysis.ipynb",
        },
        {
            "paper_section": "Abstract/Results 4.2",
            "claim_id": "bucket_framing_excluded",
            "claim_text_short": "Conservative unresolved rate on full-reasoning prompt-layer cells plus Claude",
            "csv_file": "bucket_summary.csv",
            "row_filter": "subset=framing_excluded_F1_plus_claude; bucket=B3_unresolved",
            "value_column": "bucket_pct",
            "filter_definition": "Same B1/B2/B3 rule on C1F1D1,C1F1D2,C2F1D1,C2F1D2,L1_claude",
            "source_raw_files": "data/full_factorial/C*F1*.jsonl; data/L1_swap/C1F1D1.jsonl",
            "generated_by": "analysis/build_canonical.py",
            "notes": "Mean bootstrap value in legacy dropF_metrics.json is 17.5%; point estimate is emitted here",
        },
        {
            "paper_section": "Results 4.3",
            "claim_id": "boundary_flip_definition",
            "claim_text_short": "Fingerprint subsets use boundary flips, not any-tier disagreement",
            "csv_file": "variant_disagreement_bridge.csv",
            "row_filter": "cell_id=<variant>",
            "value_column": "boundary_flip_n,boundary_flip_pct",
            "filter_definition": "sign(anchor_score) != sign(variant_score)",
            "source_raw_files": "data/full_factorial/*.jsonl; data/L1_swap*/C1F1D1.jsonl",
            "generated_by": "analysis/build_canonical.py",
            "notes": "Resolves PW #C1 definition collision",
        },
        {
            "paper_section": "Results 4.3",
            "claim_id": "d_axis_coding_only",
            "claim_text_short": "D-axis any-tier disagreement persists on coding-only prompts",
            "csv_file": "d_axis_coding_only.csv",
            "row_filter": "subset=coding",
            "value_column": "any_tier_pct",
            "filter_definition": "C1F1D1 != C1F1D2 among coding prompts",
            "source_raw_files": "data/full_factorial/C1F1D1.jsonl; data/full_factorial/C1F1D2.jsonl; data/preprocessed_pairs.jsonl",
            "generated_by": "analysis/build_canonical.py",
            "notes": "Bounds D2 mixed-input concern",
        },
        {
            "paper_section": "Results/Appendix J",
            "claim_id": "fixed_style_control_claude",
            "claim_text_short": "Claude divergence survives fixed anchor style coefficients",
            "csv_file": "fixed_style_control_ablation.csv",
            "row_filter": "cell_id=L1_claude",
            "value_column": "delta_vs_anchor_fixed_pp",
            "filter_definition": "Style-control adjustment uses C1F1D1 length/md coefficients for every cell",
            "source_raw_files": "results/style_features.csv (frozen from prior 10_analysis run); data/preprocessed_pairs.jsonl; results/canonical_v1/verdict_scores_wide.csv",
            "generated_by": "analysis/build_canonical.py",
            "notes": "Addresses Style Control refit concern; uses exact generated style_features.csv from original Arena-Hard extractor",
        },
        {
            "paper_section": "Appendix N",
            "claim_id": "magnitude_shift_breakdown",
            "claim_text_short": "Separates same-side magnitude shifts from strict preference reversals",
            "csv_file": "magnitude_shift_breakdown.csv",
            "row_filter": "cell_id=<variant>",
            "value_column": "same_side_magnitude_shift_pct,strong_weak_shift_pct,strict_reversal_pct",
            "filter_definition": "same sign with changed absolute score; plus strict sign reversal",
            "source_raw_files": "results/canonical_v1/verdict_scores_wide.csv",
            "generated_by": "analysis/build_canonical.py",
            "notes": "Addresses magnitude-threshold sensitivity traceability",
        },
        {
            "paper_section": "Results/Figure 1",
            "claim_id": "repeat_floor_boundary_change",
            "claim_text_short": "Identical baseline repeats change winner or tie in 20.0% of paired cases",
            "csv_file": "repeat_floor.csv",
            "row_filter": "single row",
            "value_column": "boundary_change_pct",
            "filter_definition": "sign(run1_score) != sign(run2_score)",
            "source_raw_files": "data/full_factorial/C1F1D1.jsonl; data/repeat_run/C1F1D1.jsonl",
            "generated_by": "analysis/build_canonical.py; analysis/repeat_floor.py",
            "notes": "Run-to-run variation under the identical C1F1D1 specification",
        },
    ]
    return pd.DataFrame(rows)


def csv_values_match(
    canonical_bytes: bytes,
    generated_bytes: bytes,
    tolerance: float = 1e-12,
) -> bool:
    """Compare CSV cells, allowing only negligible floating-point drift."""
    canonical_rows = list(csv.reader(io.StringIO(canonical_bytes.decode("utf-8"))))
    generated_rows = list(csv.reader(io.StringIO(generated_bytes.decode("utf-8"))))
    if len(canonical_rows) != len(generated_rows):
        return False

    for canonical_row, generated_row in zip(canonical_rows, generated_rows):
        if len(canonical_row) != len(generated_row):
            return False
        for canonical_value, generated_value in zip(canonical_row, generated_row):
            if canonical_value == generated_value:
                continue
            try:
                canonical_number = float(canonical_value)
                generated_number = float(generated_value)
            except ValueError:
                return False
            if math.isnan(canonical_number) and math.isnan(generated_number):
                continue
            if not math.isclose(
                canonical_number,
                generated_number,
                rel_tol=tolerance,
                abs_tol=tolerance,
            ):
                return False
    return True


def write_verified_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    """Write new outputs or verify them against the shipped canonical bytes.

    Linear algebra libraries can differ in the final floating-point bit across
    platforms. When an existing canonical CSV differs only within the strict
    tolerance above, retain its exact bytes. Any structural, textual, or larger
    numerical difference fails the build.
    """
    generated = {
        name: df.to_csv(index=False).encode("utf-8")
        for name, df in outputs.items()
    }

    for name, generated_bytes in generated.items():
        target = OUT / name
        if not target.exists():
            continue
        canonical_bytes = target.read_bytes()
        if canonical_bytes == generated_bytes:
            continue
        if not csv_values_match(canonical_bytes, generated_bytes):
            raise RuntimeError(
                f"regenerated values differ materially from shipped canonical file: {target}"
            )

    for name, df in outputs.items():
        target = OUT / name
        generated_bytes = generated[name]
        if not target.exists() or target.read_bytes() == generated_bytes:
            target.write_bytes(generated_bytes)
            status = "wrote"
        else:
            status = "verified; preserved canonical bytes for"
        print(f"{status} {target} ({len(df)} rows)")


def main() -> None:
    meta = pd.read_json(ROOT / "data" / "preprocessed_pairs.jsonl", lines=True)
    long, wide = load_scores()
    style = compute_style_features(meta)
    control, fixed = style_control_tables(wide, style)
    bridge = disagreement_bridge(wide)
    buckets, bucket_assignments = bucket_summary(wide)
    magnitudes = magnitude_shift_breakdown(bridge)
    d_check = d_axis_coding_only(wide.reset_index(), meta)
    lang_sub = language_by_subcategory(wide.reset_index(), meta)
    tests, logistic = axis_fingerprint_tests(wide, meta)
    claude_len = claude_output_lengths(long)
    repeat_floor = repeat_floor_table()
    manifest = write_manifest()

    outputs = {
        "judgment_scores_long.csv": long,
        "verdict_scores_wide.csv": wide.reset_index(),
        "style_features.csv": style,
        "controlled_winrates.csv": control,
        "fixed_style_control_ablation.csv": fixed,
        "bucket_summary.csv": buckets,
        "bucket_assignments.csv": bucket_assignments,
        "variant_disagreement_bridge.csv": bridge,
        "magnitude_shift_breakdown.csv": magnitudes,
        "d_axis_coding_only.csv": d_check,
        "language_by_subcategory_boundary_flips.csv": lang_sub,
        "axis_fingerprint_feature_tests.csv": tests,
        "axis_feature_logistic_models.csv": logistic,
        "claude_output_length_check.csv": claude_len,
        "repeat_floor.csv": repeat_floor,
        "manifest.csv": manifest,
    }
    write_verified_outputs(outputs)


if __name__ == "__main__":
    main()
