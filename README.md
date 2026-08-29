# Specification-sensitivity experiments for LLM judges

This release evaluates how the outcomes of an LLM-as-judge benchmark change under controlled variations in judge prompt, reasoning instruction, decision scope, and judge model, including an unchanged-condition repeat that measures run-to-run variation. It contains the experiment runners, frozen raw judge records, exploratory notebooks, canonical analysis definitions, paper figures, and the versioned tables used for paper-facing results.

## Python 3.12 setup

Create and activate a clean environment, then install the pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The offline canonical build does not require API credentials. The experiment runners require the relevant provider credential to be supplied through the environment or an optional local `.env` file; `.env` files are intentionally excluded from this release.

## Regenerating every paper-facing number

Run this command from the release root:

```bash
python analysis/build_canonical.py
```

All reported figures and numerical claims, including the unchanged-condition repeat floor, trace to `results/canonical_v1/manifest.csv` and the versioned CSV files under `results/canonical_v1/`. They do not trace to the legacy exploratory CSV files formerly written at the top level of `results/`. The only retained top-level result is the frozen `results/style_features.csv` input described below.

The builder recomputes every output and checks it against the shipped canonical CSV. Numerical libraries can differ in the final floating-point bit across platforms, so the builder preserves the shipped canonical bytes when every generated value agrees within `1e-12`; it fails on any structural, textual, or larger numerical difference.

## Regenerating the paper figures

Run this command from the release root after regenerating or verifying the canonical outputs:

```bash
python analysis/make_figures.py
```

This reads `results/canonical_v1/verdict_scores_wide.csv` and writes `figures/fig1.pdf`, `figures/fig2.pdf`, and PNG previews. The release tracks the two paper-facing PDF files.

## External Arena-Hard dependencies

The Arena-Hard dataset and Arena-Hard repository are external dependencies. They are not vendored as upstream repositories in this release. To obtain them:

```bash
git lfs install
git clone https://huggingface.co/datasets/lmarena-ai/arena-hard-auto arena-hard-data
cp -R arena-hard-data/data/arena-hard-v2.0 data/
git clone https://github.com/lmarena/arena-hard-auto.git arena-hard-repo
```

The frozen `data/preprocessed_pairs.jsonl` and raw judge outputs needed by the canonical build are already included. The external dataset and repository are needed to repeat the initial preprocessing notebook and workflows that depend on Arena-Hard utilities.

## File map

| File | Role and output |
| --- | --- |
| `_prompts.py` | Defines the controlled prompt cells used by the experiment runners; no direct output. |
| `_pipeline.py` | Implements shared OpenAI judge calls, verdict parsing, resumable execution, and JSONL records; outputs are written by the runners. |
| `run_test.py` | Runs a five-prompt smoke test and writes `data/test_run/C1F1D1.jsonl`. |
| `run_full_factorial.py` | Runs the eight prompt-layer cells and writes `data/full_factorial/<cell>.jsonl`. |
| `run_l1_swap.py` | Runs the Claude judge-model swap and writes `data/L1_swap/C1F1D1.jsonl`. |
| `retry_l1_failed.py` | Retries missing or unparsable Claude records and rewrites `data/L1_swap/C1F1D1.jsonl` after making a timestamped backup. |
| `run_repeat_baseline.py` | Repeats the unchanged GPT-4.1 baseline specification and writes `data/repeat_run/C1F1D1.jsonl`; running it incurs API cost. |
| `analysis/_canonical_defs.py` | Defines cell mappings, score conversion, disagreement filters, and the B1/B2/B3 bucket rule; no direct output. |
| `analysis/build_canonical.py` | Regenerates all sixteen versioned CSV files under `results/canonical_v1/`, including the repeat floor and `manifest.csv`. |
| `analysis/repeat_floor.py` | Reports the unchanged-condition noise floor and writes `results/canonical_v1/repeat_floor.csv`. |
| `analysis/make_figures.py` | Recomputes Figures 1 and 2 from `results/canonical_v1/verdict_scores_wide.csv` and writes PDF and PNG outputs under `figures/`. |
| `01_data_load.ipynb` | Fetches and validates Arena-Hard inputs and writes `data/preprocessed_pairs.jsonl`. |
| `02_pilot.ipynb` | Runs and checks the four-cell pilot, using `data/pilot/<cell>.jsonl`. |
| `03_full_factorial.ipynb` | Documents the full-factorial experiment design; `run_full_factorial.py` is the resumable production runner. |
| `04_L1_swap_claude.ipynb` | Documents the Claude anchor-cell swap; `run_l1_swap.py` is the resumable production runner. |
| `05_L1_swap_gemini.ipynb` | Runs the Gemini Pro anchor-cell swap and writes `data/L1_swap_v2/C1F1D1.jsonl`. |
| `05b_L1_swap_gemini_flash.ipynb` | Runs the Gemini Flash anchor-cell swap and writes `data/L1_swap_v2_flash_16k/C1F1D1.jsonl`. |
| `10_analysis.ipynb` | Performs exploratory aggregation, bucket, bootstrap, variance, and style-control analyses; its top-level `results/` outputs are legacy, except for the frozen style-feature input. |
| `10b_variance_without_F.ipynb` | Produces exploratory dropped-framing bucket, strict-reversal, and metrics outputs under `results/`. |
| `10c_axis_fingerprints.ipynb` | Produces exploratory axis-by-style, margin, prompt-feature, and regime tables under `results/`. |
| `10d_robustness_checks.ipynb` | Produces exploratory variance-matched, bootstrap, invalid-verdict, Jaccard, and robustness outputs under `results/`. |
| `10e_anchor_swap.ipynb` | Produces anchor-swap disagreement and ranking tables under `results/anchor_swap/`. |
| `11_inter_judge_agreement.ipynb` | Produces `results/inter_judge_agreement.csv` and updates the exploratory metrics summary. |
| `data/` | Frozen preprocessed inputs, original raw judge records, and the unchanged-condition repeat consumed by the canonical build. |
| `results/style_features.csv` | Frozen style-feature input imported by the canonical build. |
| `results/canonical_v1/` | Versioned paper-facing outputs and claim-level manifest. |
| `figures/fig1.pdf` | Paper Figure 1: aggregate-level changes and case-level boundary changes. |
| `figures/fig2.pdf` | Paper Figure 2: sensitivity counts and change rates by baseline margin. |

## Known limitations of this release

`results/style_features.csv` is frozen output from a feature extractor that is no longer available. `analysis/build_canonical.py` imports this file rather than recomputing it. The style-controlled analysis therefore cannot be regenerated from source by anyone; only the downstream calculations from the frozen feature table can be rerun.
