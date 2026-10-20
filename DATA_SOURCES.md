# Data sources and attribution

The repository combines original analysis code, transformed upstream benchmark inputs, and model-generated experimental records. The root [MIT licence](LICENSE) covers the original code and documentation. It does not replace upstream terms or assert ownership of text supplied by others.

## Arena-Hard inputs and reference implementation

`data/preprocessed_pairs.jsonl` contains the study's frozen 500-prompt input: Arena-Hard questions and metadata, GPT-4.1 mini candidate answers, and o3-mini baseline answers. `01_data_load.ipynb` documents preprocessing from the upstream `data/arena-hard-v2.0/` tree. Default judging instructions and benchmark conventions in `_prompts.py` also derive from Arena-Hard.

| Source | Recorded reference |
| --- | --- |
| [Arena-Hard dataset](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto) | Revision `15f3746`; upstream `data/arena-hard-v2.0/` |
| [Arena-Hard reference code](https://github.com/lmarena/arena-hard-auto) | Commit [`196f6b826783b3da7310e361a805fa36f0be83f3`](https://github.com/lmarena/arena-hard-auto/commit/196f6b826783b3da7310e361a805fa36f0be83f3) |

These are the upstream references recorded for the study. The [official dataset history](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/commits/main) lists `15f3746`; the code's full commit was resolved from `196f6b8`. The release's own frozen-file hashes are in `data/manifest.csv`. The packaged transformed input has not been claimed to be a byte-for-byte copy of an upstream file.

The [upstream dataset card](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/blob/main/README.md) declares Apache-2.0. The [reference code's licence at the recorded commit](https://github.com/lmarena/arena-hard-auto/blob/196f6b826783b3da7310e361a805fa36f0be83f3/LICENSE) is Apache-2.0; an unchanged copy is supplied at [third_party/arena-hard-LICENSE](third_party/arena-hard-LICENSE). Upstream material retains that licence and its attribution. This records the upstream declaration without asserting separate rights over every embedded prompt, answer, quotation or code fragment.

Please also credit Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E. Gonzalez and Ion Stoica, [*From Crowdsourced Data to High-Quality Benchmarks: Arena-Hard and BenchBuilder Pipeline*](https://arxiv.org/abs/2406.11939), the benchmark work identified by the upstream dataset card.

The upstream repositories are not vendored. Repeating input preparation requires retrieving those sources at the recorded revisions; it is separate from rebuilding this release's numerical outputs from the included inputs.

## Experimental judgments

The JSONL files under `data/full_factorial/`, `data/repeat_run/`, `data/L1_swap*/` and `data/october_rerun/` are frozen judge responses collected for this study. They contain generated explanations and verdicts, sometimes quoting the benchmark inputs. The October files additionally record the served model, system fingerprint and verdict-token log probabilities. Invalid verdicts are retained.

These files are experimental evidence, not an independently curated ground-truth dataset. Their inclusion supplies the provenance needed to inspect and recompute the reported analyses. The MIT grant concerns the original software; it does not purport to relicense provider-generated text or third-party material embedded in these records. Preserve their source attribution and distinguish new API outputs from these frozen observations.

`data/L1_swap/C1F1D1_unrecoverable.jsonl` is an intentionally retained empty marker. Its zero-byte hash and record count are listed alongside the other JSONL inputs.

## Derived tables and figures

The released scripts derive `results/canonical_v1/`, `results/october_rerun/` and their corresponding figures from the frozen inputs. The manifests identify the source files and producing scripts. `results/style_features.csv` is an exception: it is preserved output from the original feature extraction workflow, whose exact extractor is unavailable. The August build reruns calculations from this table rather than regenerating its features.

The full Arena-Hard leaderboard's joint Bradley–Terry/style-controlled fit requires upstream judgments for other models. It is not part of either offline build in this release. The optional [external check procedure](docs/EXTERNAL_STYLE_CHECK.md) accompanies `analysis/arena_hard_style_check_prepare.py` and the unchanged [historical report](results/external/arena_hard_style_check.md); the estimator was not rerun for this candidate.
