"""Unchanged-condition repeat of the baseline cell.

Re-runs C1F1D1 (Arena-Hard criteria, reasoning required, no scope) with the same
judge model, temperature and answer-order protocol as the original run, into a
separate output directory. Comparing the two runs gives the run-to-run
nondeterminism floor that the original design could not measure.

Nothing about the specification changes. Only the run does.
"""
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from _pipeline import run_cell_batch_parallel

EXPERIMENT_ROOT = Path(__file__).resolve().parent
CELL = "C1F1D1"
OUTPUT = EXPERIMENT_ROOT / "data" / "repeat_run" / f"{CELL}.jsonl"
MAX_WORKERS = 16

load_dotenv(EXPERIMENT_ROOT / ".env")
client = OpenAI()

pairs = pd.read_json(EXPERIMENT_ROOT / "data" / "preprocessed_pairs.jsonl", lines=True)
print(f"loaded {len(pairs)} pairs; writing to {OUTPUT}", flush=True)

run_cell_batch_parallel(
    client,
    cell_id=CELL,
    pairs=pairs,
    output_path=OUTPUT,
    resume=True,
    max_workers=MAX_WORKERS,
)
print("done", flush=True)
