# Behind Aggregate Scores: reproducibility release

Code and frozen data for **Behind Aggregate Scores: What Systematically Varying Arena-Hard's LLM Judge Reveals**, by Louise Yorke, NeurIPS 2026 JUDGe workshop. [Paper](https://openreview.net/forum?id=knP38wcoPO) · [Citation metadata](CITATION.cff) · [Data sources](DATA_SOURCES.md) · [Validation report](VALIDATION.md)

This version 1.0.0 camera-ready candidate contains two experiment waves. The October analysis is the current camera-ready reproduction path; the August review release is preserved separately.

| Wave | Raw records | Results | Figures |
| --- | --- | --- | --- |
| October rerun | `data/october_rerun/` | `results/october_rerun/` | `figures/october/` |
| Original experiment and repeat, released in August | `data/full_factorial/`, `data/repeat_run/`, `data/L1_swap*/` | `results/canonical_v1/` | `figures/fig1.pdf`–`fig3.pdf` |

## Setup and offline reproduction

Use Python 3.12 and the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Recompute and verify the October tables, then render the three figures and six table PDFs:

```bash
python analysis/build_october.py
python analysis/make_october_figures_tables.py
```

The builder runs the step analyses in dependency order and checks their results against the 13 shipped October CSVs. `--check-only` verifies without exporting outputs; `--output-dir PATH` exports verified tables elsewhere. The figure script writes PDFs and PNG previews to `figures/october/`; its `--output-dir PATH` option changes that destination. These commands make no API calls.

The equivalent Make targets are `make october`, `make legacy`, and `make all`. `make verify` checks the frozen-data inventory and both waves' numerical outputs; `make figures` renders both waves' figures; `make test` runs the offline release-interface tests. Set `PYTHON=...` when using another interpreter.

The retained August build is:

```bash
python analysis/build_canonical.py
python analysis/make_figures.py
PYTHONPATH=analysis python analysis/make_exploratory_figure.py
```

It verifies 19 canonical CSVs. Its original claim map is `results/canonical_v1/manifest.csv`. The supplementary verifiable-item checks run with `PYTHONPATH=analysis python analysis/exploratory_checks.py`; those checks require a C++17 compiler available as `c++` and write `results/exploratory_checks.md`.

## What the October records contain

The same 500 prompt IDs and frozen answer pairs are evaluated in all eight prompt configurations twice, with a third run of the default configuration: **17 configuration-runs, 8,500 paired records and 17,000 order-specific judgments**. Every served-model field records `gpt-4.1-2025-04-14`. Fingerprints are present for all judgments, with 37 distinct values; a pinned model name therefore does not mean a single observed backend fingerprint.

Four order-specific verdicts failed parsing: two in default run 1, one in default run 2, and one in the description configuration's run 2. Analyses require both answer orders to parse. The three default runs have 497 common complete prompts; comparisons with the description configuration use 496, and the other six configuration comparisons use 497. Analyses requiring all 17 runs use 496 prompts. Invalid records remain in the frozen data.

`data/manifest.csv` records input hashes and record counts. `results/october_rerun/manifest.csv` maps the October outputs to their producers and inputs. The step scripts document the comparison rules and use 20,000 prompt-level bootstrap resamples with seed 20260830 where intervals are reported. Tables distinguish the default configuration's repeat rate from the comparison configuration's own repeat rate.

## File map

| Path | Purpose |
| --- | --- |
| `_prompts.py`, `_pipeline.py` | Prompt configurations, API calls, verdict parsing and raw-record format |
| `run_october_rerun.py` | Optional new paid October-design experiment |
| `analysis/build_october.py`, `analysis/october_step*.py` | Offline October orchestration and statistical analyses |
| `analysis/make_october_figures_tables.py` | October figures and table images |
| `analysis/validate_release.py` | Frozen-input and release-inventory checks |
| `analysis/_canonical_defs.py`, `analysis/repeat_floor.py` | Shared score conversion and original repeat analysis |
| `data/preprocessed_pairs.jsonl` | Frozen prompts and the two model answers compared |
| `results/style_features.csv` | Frozen input to the original style-controlled analysis |
| `01_*.ipynb`–`11_*.ipynb`, other `run_*.py` | Retained preprocessing, original experiment and exploratory workflows |

## Running new API experiments

Rebuilding the released results needs no credentials. New experiments incur provider charges and need a provider API key in the environment or a local `.env` file. The October runner uses the pinned model and defaults to `new_api_runs/october/`, separate from the frozen release inputs:

```bash
python run_october_rerun.py --runs run1 run2
python run_october_rerun.py --runs run3 --cells C1F1D1
```

Use `--output-root PATH` to choose another destination. The runner rejects destinations within frozen `data/`; new API calls need not reproduce the recorded judgments. Historical runners and notebooks retain their original experiment paths and should be used in a separate working copy for new data collection.

## Scope and limitations

- The original style-feature extractor is unavailable. The August build preserves `results/style_features.csv` and reproduces downstream calculations from that frozen table; it does not recreate the features from source.
- Historical notebooks are exploratory records. The two build scripts and their versioned tables define the supported reproduction paths; the original notebook bootstraps are not uniformly seeded.
- Upstream dataset retrieval and Arena-Hard leaderboard fitting are separate from these offline builds. The optional [external style-check procedure](docs/EXTERNAL_STYLE_CHECK.md) provides the preparation helper and [historical captured report](results/external/arena_hard_style_check.md); the external estimator has not been rerun for this candidate. See [DATA_SOURCES.md](DATA_SOURCES.md) for upstream revisions and provenance.
- Original code and documentation use the [MIT licence](LICENSE). Upstream Arena-Hard material retains its own attribution and licence; model-generated records and embedded third-party text are described separately in [DATA_SOURCES.md](DATA_SOURCES.md).
