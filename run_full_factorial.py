"""Standalone runner for the full L2 factorial. Decoupled from the notebook so
it survives Codespace idle / browser close.

Usage:
    cd llm-judge-release
    nohup python run_full_factorial.py > run_full_factorial.log 2>&1 &
    tail -f run_full_factorial.log

Resume support: if the script dies (timeout, network blip, kernel crash),
re-running picks up where it left off — already-judged uids per cell are skipped
via `resume=True` in run_cell_batch_parallel.

Scope: 500 prompts × 8 L2 cells × 2 rounds = 8,000 judge calls at gpt-4.1.
Estimated runtime: ~3 hours with max_workers=16. Estimated cost: ~£35-45 with
prompt caching.
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from _pipeline import run_cell_batch_parallel
from _prompts import L2_CELLS

# Load API keys from an optional release-root .env file.
EXPERIMENT_ROOT = Path(__file__).resolve().parent
load_dotenv(EXPERIMENT_ROOT / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load preprocessed pairs from notebook 01
pairs = pd.read_json(EXPERIMENT_ROOT / "data" / "preprocessed_pairs.jsonl", lines=True)
print(f"Loaded {len(pairs)} pairs", flush=True)

OUTPUT_DIR = EXPERIMENT_ROOT / "data" / "full_factorial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Track timing across cells
overall_start = time.time()
cell_durations = []

print(f"\n{'='*60}", flush=True)
print(f"FULL FACTORIAL — start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
print(f"{len(L2_CELLS)} cells × {len(pairs)} prompts × 2 rounds = {len(L2_CELLS) * len(pairs) * 2} judge calls", flush=True)
print(f"{'='*60}\n", flush=True)

# Run all 8 L2 cells sequentially; within each cell, max_workers parallel calls.
for i, cell_id in enumerate(L2_CELLS, 1):
    output_path = OUTPUT_DIR / f"{cell_id}.jsonl"
    cell_start = time.time()
    print(f"\n--- [{i}/{len(L2_CELLS)}] {cell_id} START {datetime.now().strftime('%H:%M:%S')} ---", flush=True)

    run_cell_batch_parallel(
        client=client,
        cell_id=cell_id,
        pairs=pairs,
        output_path=output_path,
        resume=True,
        max_workers=6,  # stays under OpenAI's 800K TPM ceiling for gpt-4.1
    )

    cell_dur = time.time() - cell_start
    cell_durations.append(cell_dur)
    avg_so_far = sum(cell_durations) / len(cell_durations)
    cells_remaining = len(L2_CELLS) - i
    est_remaining = timedelta(seconds=int(avg_so_far * cells_remaining))
    elapsed_total = timedelta(seconds=int(time.time() - overall_start))

    print(
        f"--- [{i}/{len(L2_CELLS)}] {cell_id} DONE in {timedelta(seconds=int(cell_dur))} "
        f"| total elapsed {elapsed_total} | est remaining {est_remaining} ---",
        flush=True,
    )

total_dur = timedelta(seconds=int(time.time() - overall_start))
print(f"\n{'='*60}", flush=True)
print(f"✓ All {len(L2_CELLS)} cells complete | total time {total_dur}", flush=True)
print(f"{'='*60}", flush=True)
