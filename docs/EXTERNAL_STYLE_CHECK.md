# External Arena-Hard style-control check

This supporting check uses Arena-Hard's joint Bradley–Terry fit with length and markdown features. The [captured 19 October report](../results/external/arena_hard_style_check.md) includes the original default configuration's style-controlled score of 48.0. It is a historical result, copied without numerical changes; the external estimator has not been rerun for this release candidate. It is separate from `make verify` and the two offline builds.

The captured intervals use **100 unseeded bootstrap draws and the 5th–95th percentiles (90%)**. The main October analyses use **20,000 seeded prompt-level draws and 95% intervals**. Do not interpret the external report's intervals as the main analysis's intervals or expect identical external bootstrap endpoints on rerun.

## Retrieve the recorded upstream revisions

Use a separate working directory with Git LFS available. The [recorded dataset revision](https://huggingface.co/datasets/lmarena-ai/arena-hard-auto/commits/main) is `15f3746`; the [reference code commit](https://github.com/lmarena/arena-hard-auto/commit/196f6b826783b3da7310e361a805fa36f0be83f3) is `196f6b826783b3da7310e361a805fa36f0be83f3`.

```bash
git clone https://github.com/lmarena/arena-hard-auto.git arena-hard-reference
git -C arena-hard-reference checkout 196f6b826783b3da7310e361a805fa36f0be83f3
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/lmarena-ai/arena-hard-auto arena-hard-data
GIT_LFS_SKIP_SMUDGE=1 git -C arena-hard-data checkout 15f3746
git -C arena-hard-data lfs pull --include='data/arena-hard-v2.0/**'
mkdir -p arena-hard-reference/data
cp -R arena-hard-data/data/arena-hard-v2.0 arena-hard-reference/data/
```

Install the upstream code's dependencies in a separate environment, following its README. They are not added to this release's pinned environment. This path needs all the upstream model answers and judgments used in the joint fit, not just this study's frozen candidate/baseline pairs. See [DATA_SOURCES.md](../DATA_SOURCES.md) for attribution and licence scope.

## Prepare and score each comparison separately

From this release's root, nominate the external checkout explicitly:

```bash
python analysis/arena_hard_style_check_prepare.py /path/to/arena-hard-reference
```

The helper makes no API calls. It **modifies the nominated checkout**, creating `ours-may`, `ours-oct-run1`, `ours-oct-run2` and `ours-oct-run3` under `data/arena-hard-v2.0/model_judgment/`. Each folder links the other models' original judgments and writes or replaces the GPT-4.1 mini judgments using this study's frozen records. Absolute and relative checkout paths are supported. `--help` describes the destination and writes no files.

Then, from the external checkout and using its environment, run the estimator once per judge folder:

```bash
for style_judge in gpt-4.1 ours-may ours-oct-run1 ours-oct-run2 ours-oct-run3; do
    python show_result.py -j "$style_judge" -f length markdown
done
```

Keep these as separate invocations: supplying all folder names to one `-j` invocation combines their judgments. The helper preserves the study's answer-order mapping: `round1` maps to `games[0]` and `round2` to `games[1]`. Retain the original captured report; save any newly computed output separately with the upstream revisions and environment used.
