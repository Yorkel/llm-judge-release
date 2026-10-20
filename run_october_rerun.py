"""Collect new runs using the October 2026 experiment's pinned configuration.

New outputs default to new_api_runs/october/run1/ and run2/ under this release.
The frozen data/ directory is read only. Resume skips prompts already saved in
each new run folder and continues from where it stopped.

Usage from the release directory:
    python run_october_rerun.py --help
    python run_october_rerun.py
    python run_october_rerun.py --runs run3 --cells C1F1D1

Set OPENAI_API_KEY in the environment or this release's .env file before
collecting data. Actual collection makes paid API calls. The default selection
is 500 prompts x 8 cells x 2 answer orders x 2 runs = 16,000 judge calls.
Use --output-root to choose another destination outside frozen data/.
Each call records the system fingerprint, model version served and verdict-tag
log probabilities. New runs are not expected to reproduce frozen verdicts.
"""

import argparse
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

from _prompts import L2_CELLS

EXPERIMENT_ROOT = Path(__file__).resolve().parent
FROZEN_DATA_ROOT = EXPERIMENT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "new_api_runs" / "october"
MODEL = "gpt-4.1-2025-04-14"  # pinned; May used the unpinned alias "gpt-4.1"
MAX_WORKERS = 6  # same as the May factorial


def run_name(value: str) -> str:
    """Accept plain folder names, never paths or traversal components."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "run names must start with a letter or digit and contain only "
            "letters, digits, underscores or hyphens"
        )
    return value


def checked_output_root(parser, output_root, runs, cells):
    """Reject destinations in frozen data, including existing symlink targets."""
    root = output_root.expanduser().resolve()
    frozen = FROZEN_DATA_ROOT.resolve()
    destinations = [root]
    destinations.extend(root / run / f"{cell}.jsonl" for run in runs for cell in cells)
    if any(path.resolve().is_relative_to(frozen) for path in destinations):
        parser.error("output destinations must be outside the frozen data/ directory")
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=run_name, default=["run1", "run2"])
    parser.add_argument("--cells", nargs="+", default=L2_CELLS, choices=L2_CELLS)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="new run destination outside frozen data/ (default: %(default)s)",
    )
    args = parser.parse_args(argv)
    output_root = checked_output_root(parser, args.output_root, args.runs, args.cells)

    # Help, imports and invalid arguments never load credentials or create a client.
    import pandas as pd
    from dotenv import load_dotenv
    from openai import OpenAI

    from _pipeline import run_cell_batch_parallel

    load_dotenv(EXPERIMENT_ROOT / ".env")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    pairs = pd.read_json(FROZEN_DATA_ROOT / "preprocessed_pairs.jsonl", lines=True)
    print(f"Loaded {len(pairs)} pairs", flush=True)
    print(f"OCTOBER RERUN — start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(
        f"{len(args.runs)} runs {args.runs} × {len(args.cells)} cells × "
        f"{len(pairs)} prompts × 2 rounds; model={MODEL}", flush=True,
    )
    print(f"Output root: {output_root}", flush=True)

    overall_start = time.time()
    for run in args.runs:
        for i, cell_id in enumerate(args.cells, 1):
            output_path = output_root / run / f"{cell_id}.jsonl"
            cell_start = time.time()
            print(
                f"\n--- {run} [{i}/{len(args.cells)}] {cell_id} "
                f"START {datetime.now().strftime('%H:%M:%S')} ---", flush=True,
            )
            run_cell_batch_parallel(
                client=client,
                cell_id=cell_id,
                pairs=pairs,
                output_path=output_path,
                resume=True,
                max_workers=MAX_WORKERS,
                model=MODEL,
                logprobs=True,
            )
            print(
                f"--- {run} [{i}/{len(args.cells)}] {cell_id} DONE in "
                f"{timedelta(seconds=int(time.time() - cell_start))} "
                f"| total elapsed {timedelta(seconds=int(time.time() - overall_start))} ---",
                flush=True,
            )

    print(
        f"\nAll runs complete | total time "
        f"{timedelta(seconds=int(time.time() - overall_start))}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
