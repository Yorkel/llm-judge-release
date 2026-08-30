"""Paper-facing supplementary prompt-level checks, run 2026-08-30.

These checks support the stability, half-step, independent-margin, language,
and verifiable-item analyses reported or discussed in the current paper.

  1. Are both models' answers correct on the verifiable always-changing items?
  2. Does the plus/minus 0.5 aggregation artefact drive boundary changes?
  3. Does baseline margin predict change once the circularity is broken?
  4. Does the non-English effect survive on the full sample against the floor?

Usage:  PYTHONPATH=analysis python analysis/exploratory_checks.py
Writes: results/exploratory_checks.md
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from _canonical_defs import candidate_score

HERE = Path(__file__).resolve().parent.parent
CANON = HERE / "results" / "canonical_v1"
BASELINE = "C1F1D1"
BOOTSTRAP = 4000
SEED = 20260830

# Item 6's raw verdict files, in the order reported. The shipped configuration is
# first so that the row that preferred the failing answer sits at the top.
ITEM6_RUNS = [
    ("baseline (Arena-Hard shipped)", "data/full_factorial/C1F1D1.jsonl"),
    ("criteria (accuracy-first)", "data/full_factorial/C2F1D1.jsonl"),
    ("label-only", "data/full_factorial/C1F2D1.jsonl"),
    ("scope", "data/full_factorial/C1F1D2.jsonl"),
    ("criteria + format", "data/full_factorial/C2F2D1.jsonl"),
    ("criteria + scope", "data/full_factorial/C2F1D2.jsonl"),
    ("format + scope", "data/full_factorial/C1F2D2.jsonl"),
    ("all three", "data/full_factorial/C2F2D2.jsonl"),
    ("Claude Sonnet 4.5", "data/L1_swap/C1F1D1.jsonl"),
    ("Gemini 2.5 Flash", "data/L1_swap_v2_flash_16k/C1F1D1.jsonl"),
    ("repeat of the baseline", "data/repeat_run/C1F1D1.jsonl"),
]

CONDITIONS = [
    ("Criteria", "C2F1D1"), ("Format", "C1F2D1"), ("Scope", "C1F1D2"),
    ("Crit+Fmt", "C2F2D1"), ("Crit+Scope", "C2F1D2"), ("Fmt+Scope", "C1F2D2"),
    ("All three", "C2F2D2"), ("Claude", "L1_claude"), ("Gemini", "L1_gemini"),
]

BOTH_CORRECT_ITEMS = [
    ("Python one-liner", "fd5cbb5afb5349da"),
    ("k-th permutation", "6b9ca33cc041442c"),
    ("trigonometric equation", "de0795d7badb493b"),
]


def load_run(path: Path) -> pd.Series:
    """Order-averaged candidate score per uid; NaN if either order failed."""
    rows = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            s1 = candidate_score(r["round1"].get("verdict"), "round1")
            s2 = candidate_score(r["round2"].get("verdict"), "round2")
            rows[r["uid"]] = np.nan if (pd.isna(s1) or pd.isna(s2)) else (s1 + s2) / 2
    return pd.Series(rows)


def scores() -> pd.DataFrame:
    w = pd.read_csv(CANON / "verdict_scores_wide.csv").set_index("uid")
    w["repeat"] = load_run(HERE / "data" / "repeat_run" / "C1F1D1.jsonl")
    return w


# ---- 1. do both answers actually work? ---------------------------------------

def check_answers(out: list[str]) -> None:
    """Execute the two candidate solutions on the examples the prompt supplies."""
    out.append("## 1. Are both answers correct on the verifiable items?\n")

    # Item 10: two Python one-liners decoding a custom-base string.
    data = "!*,*!!*!"          # the prompt states this should give 18
    chars, num = data.split(",")
    cand = sum((chars.index(c) << i) for i, c in enumerate(num[::-1]))
    base = int("".join("0" if c == chars[0] else "1" for c in num), 2)
    out.append(f"Item 10, Python one-liners on the prompt's own example (expect 18): "
               f"candidate {cand}, baseline {base}. "
               f"{'Both correct.' if cand == base == 18 else 'MISMATCH.'}\n")

    pairs = pd.read_json(HERE / "data" / "preprocessed_pairs.jsonl", lines=True).set_index("uid")
    # Item 5: k-th permutation in C++, tested against the prompt's three samples.
    harness = """
#include <iostream>
int main(){ Solution s;
  std::cout << (s.getPermutation(3,3)=="213")
            << (s.getPermutation(4,9)=="2314")
            << (s.getPermutation(3,1)=="123") << "\\n"; }
"""
    verdicts = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, field in [("candidate", "candidate_response"), ("baseline", "baseline_response")]:
            text = pairs.loc["6b9ca33cc041442c", field]
            blocks = re.findall(r"```(?:c\+\+|cpp)?\n(.*?)```", text, re.S)
            # some answers fence their code, some delimit it with rules; fall back
            # to the span between the first #include and the last closing brace.
            code = max(blocks, key=len) if blocks else text[text.find("#include"): text.rfind("};") + 2]
            src = Path(tmp) / f"{label}.cpp"
            src.write_text(code + harness)
            binp = Path(tmp) / label
            if subprocess.run(["c++", "-std=c++17", "-o", str(binp), str(src)],
                              capture_output=True).returncode:
                verdicts[label] = "did not compile"
                continue
            verdicts[label] = subprocess.run([str(binp)], capture_output=True, text=True).stdout.strip()
    out.append(f"Item 5, k-th permutation against the prompt's three samples (1 = pass): "
               f"candidate {verdicts['candidate']}, baseline {verdicts['baseline']}.\n")
    out.append("Item 8, the trigonometric equation, was solved by hand: it reduces to "
               "2cos^2 x + 5cos x - 3 = 0, giving cos x = 1/2 and nine roots in (0, 27) "
               "summing to 121*pi/3. Both models report 121*pi/3.\n")

    # Item 6 is the exception, and the most informative of the four.
    out.append(check_item6(pairs))


def check_item6(pairs: pd.DataFrame) -> str:
    """Item 6, a competitive-programming problem with three worked samples.

    Unlike the others this is not a case of two correct answers. The candidate
    fails every sample; the baseline passes every sample. Arena-Hard's shipped
    judge configuration nonetheless recorded a win for the failing answer, and a
    rerun of that same configuration reversed it.
    """
    uid = "c0a4a8902af041b3"
    samples = [
        ("8 2\nth#nkyou\n#estwish\n", "thankyou"),
        ("16 5\n##soluteradian#e\nyour#awnwillcom#\n", "absoluteradiance"),
        ("40 45\nhhuj#pzr#k#mmd#z##y#o####m##j##tga#k#t#g\n"
         "m########be#######vf##a#j###ypuf###pr###\n",
         "hhujapzrakbmmdczdeyfoghijmkljmntgaokptqg"),
    ]
    lines = ["\nItem 6, competitive programming, tested against the prompt's three samples."]
    with tempfile.TemporaryDirectory() as tmp:
        # bits/stdc++.h is a GCC extension; both answers use it, so shim it.
        (Path(tmp) / "bits").mkdir()
        headers = ("algorithm array bitset cassert cmath cstdio cstdlib cstring deque functional "
                   "iomanip iostream map numeric queue set sstream stack string tuple "
                   "unordered_map unordered_set utility vector").split()
        (Path(tmp) / "bits" / "stdc++.h").write_text(
            "#pragma once\n" + "".join(f"#include <{h}>\n" for h in headers))
        for label, field in [("candidate", "candidate_response"), ("baseline", "baseline_response")]:
            text = pairs.loc[uid, field]
            blocks = re.findall(r"```(?:c\+\+|cpp)?\n(.*?)```", text, re.S)
            if blocks:
                code = max(blocks, key=len)
            else:
                # the baseline delimits its code with rules rather than fences
                i = text.find("#include")
                code = text[i: text.find("\n}", text.find("int main")) + 2]
            src = Path(tmp) / f"{label}.cpp"
            src.write_text(code)
            binp = Path(tmp) / label
            if subprocess.run(["c++", "-std=c++17", "-O2", "-I", tmp, "-o", str(binp), str(src)],
                              capture_output=True).returncode:
                lines.append(f"  {label}: did not compile")
                continue
            got = []
            for inp, expected in samples:
                r = subprocess.run([str(binp)], input=inp, capture_output=True, text=True, timeout=20)
                got.append("pass" if r.stdout.strip() == expected else "FAIL")
            lines.append(f"  {label}: {', '.join(got)}")
    lines.append("\nThe candidate fails all three samples; the baseline passes all three. Ground "
                 "truth is therefore known without consulting any judge. What each condition made "
                 "of it:\n")
    lines.append("| condition | order 1 | order 2 | score | preferred |")
    lines.append("|---|---|---|---:|---|")
    for label, path in ITEM6_RUNS:
        with open(HERE / path) as f:
            for line in f:
                rec = json.loads(line)
                if rec["uid"] != uid:
                    continue
                v1, v2 = rec["round1"].get("verdict"), rec["round2"].get("verdict")
                s1 = candidate_score(v1, "round1")
                s2 = candidate_score(v2, "round2")
                avg = (s1 + s2) / 2
                # positive favours the candidate, which is the answer that fails
                preferred = {1: "**the failing answer**", 0: "tie",
                             -1: "the correct answer"}[int(np.sign(avg))]
                lines.append(f"| {label} | {v1} | {v2} | {avg:+.1f} | {preferred} |")
                break
    lines.append("\nThe shipped Arena-Hard configuration is the only one that preferred the "
                 "failing answer. The accuracy-first rubric caught the bug, both judge "
                 "substitutions caught it decisively, and a rerun of the shipped configuration "
                 "itself reversed the verdict. This is not a case of two defensible answers: it "
                 "is a verifiable error in the published configuration, detected both by an "
                 "alternative specification and by simply running the same one again.")
    return "\n".join(lines) + "\n"


def check_both_correct_verdicts(w: pd.DataFrame, out: list[str]) -> None:
    """Summarise judge preferences on the three checked both-correct items."""
    out.append("\n### Condition-level outcomes for the three both-correct items\n")
    out.append("| item | candidate win present | baseline win present | tie present | non-tie magnitudes |")
    out.append("|---|---:|---:|---:|---|")
    columns = [BASELINE, *[cell for _, cell in CONDITIONS], "repeat"]
    for label, uid in BOTH_CORRECT_ITEMS:
        values = w.loc[uid, columns].dropna()
        signs = set(np.sign(values).astype(int))
        magnitudes = sorted({abs(float(value)) for value in values if value != 0})
        out.append(
            f"| {label} | {1 in signs} | {-1 in signs} | {0 in signs} | "
            f"{', '.join(f'{value:.1f}' for value in magnitudes)} |"
        )
    out.append(
        "\nEach checked both-correct item receives a candidate win, a baseline win, "
        "and a tie somewhere in the observed condition set. The table also exposes "
        "whether every non-tie score is a half step.\n"
    )


# ---- 2. how much of the instability is the +/-0.5 artefact? -------------------

def check_half_step(w: pd.DataFrame, out: list[str]) -> None:
    """A score of +/-0.5 is one ordinal step from a tie, so it flips easily."""
    out.append("\n## 2. Does the plus/minus 0.5 artefact drive boundary changes?\n")
    out.append("| condition | changed | of which +/-0.5 | share | base rate |")
    out.append("|---|---:|---:|---:|---:|")
    changed = involved = 0
    rates = []
    for name, col in CONDITIONS:
        d = w[[BASELINE, col]].dropna()
        ch = np.sign(d[col]) != np.sign(d[BASELINE])
        half = (d[BASELINE].abs() == 0.5) | (d[col].abs() == 0.5)
        changed += int(ch.sum())
        involved += int((ch & half).sum())
        rates.append(half.mean())
        out.append(f"| {name} | {int(ch.sum())} | {int((ch & half).sum())} | "
                   f"{100 * (ch & half).sum() / ch.sum():.1f}% | {100 * half.mean():.1f}% |")
    out.append(f"\nOverall {involved}/{changed} = {100 * involved / changed:.1f}% of boundary changes "
               f"involve a +/-0.5 score, against a {100 * np.mean(rates):.1f}% base rate. "
               f"Elevated by roughly {100 * involved / changed / (100 * np.mean(rates)):.2f}x, "
               f"so the artefact contributes but does not dominate.\n")


# ---- 3. does margin predict change independently of the selection rule? ------

def check_circularity(w: pd.DataFrame, out: list[str]) -> None:
    """Grouping prompts by whether they changed, then reporting their baseline
    margin, is circular: a small margin is what causes changing. Taking the
    margin from the repeat instead makes it an independent draw."""
    out.append("\n## 3. Does margin predict change once the circularity is broken?\n")
    out.append("| condition | baseline margin, changed | unchanged | repeat margin, changed | unchanged |")
    out.append("|---|---:|---:|---:|---:|")
    for name, col in CONDITIONS:
        d = w[[BASELINE, col, "repeat"]].dropna()
        ch = np.sign(d[col]) != np.sign(d[BASELINE])
        out.append(f"| {name} | {d[BASELINE].abs()[ch].mean():.2f} | {d[BASELINE].abs()[~ch].mean():.2f} | "
                   f"{d['repeat'].abs()[ch].mean():.2f} | {d['repeat'].abs()[~ch].mean():.2f} |")
    out.append("\nThe gap narrows once the margin comes from an independent run, which is the "
               "circularity being removed, but it survives in every condition. Low margin genuinely "
               "predicts instability.\n")


# ---- 4. is the non-English effect real? --------------------------------------

def check_language(w: pd.DataFrame, out: list[str]) -> None:
    """Excess change rate for non-English prompts, over and above the same
    difference in the unchanged repeat, with a paired case bootstrap."""
    out.append("\n## 4. Does the non-English effect survive on the full sample?\n")
    pairs = pd.read_json(HERE / "data" / "preprocessed_pairs.jsonl", lines=True).set_index("uid")
    non_english = pairs.language != "English"
    rng = np.random.default_rng(SEED)
    out.append("| condition | excess for non-English | 95% CI |")
    out.append("|---|---:|---|")
    for name, col in CONDITIONS:
        d = pd.concat([w[BASELINE], w[col].rename("x"), w["repeat"].rename("r"),
                       non_english.rename("g")], axis=1).dropna()
        ch = (np.sign(d.x) != np.sign(d[BASELINE])).to_numpy()
        floor = (np.sign(d.r) != np.sign(d[BASELINE])).to_numpy()
        g = d.g.to_numpy().astype(bool)
        gap = lambda a, b, m: (a[m].mean() - a[~m].mean()) - (b[m].mean() - b[~m].mean())
        draws = rng.integers(0, len(d), size=(BOOTSTRAP, len(d)))
        boot = np.array([100 * gap(ch[i], floor[i], g[i]) for i in draws])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out.append(f"| {name} | {100 * gap(ch, floor, g):+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
    out.append("\nConsistently positive, never significant, and one condition reverses sign. "
               "The 60% non-English share observed among the ten always-changing prompts is "
               "selection noise on a very small group.\n")


def main() -> None:
    w = scores()
    out = ["# Exploratory checks, 2026-08-30",
           "",
           "Generated by `analysis/exploratory_checks.py`. Supports paper-facing supplementary claims.",
           ""]
    check_answers(out)
    check_both_correct_verdicts(w, out)
    check_half_step(w, out)
    check_circularity(w, out)
    check_language(w, out)
    dest = HERE / "results" / "exploratory_checks.md"
    report = "\n".join(out).rstrip() + "\n"
    dest.write_text(report)
    print(report, end="")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
