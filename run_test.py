"""Smoke test for the standalone runner.

5 prompts × 1 cell × 2 rounds = 10 judge calls, ~£0.05, <2 minutes.

Validates that the runner script imports cleanly, that nohup logging works,
that prompt caching kicks in, and that the JSONL output writes correctly.

Usage:
    cd llm-judge-release
    nohup python run_test.py > run_test.log 2>&1 &
    tail -f run_test.log

Inspect output afterwards:
    head data/test_run/C1F1D1.jsonl

If the test runs cleanly, kick off run_full_factorial.py. If it errors, debug
before scaling.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from _pipeline import run_cell_batch_parallel

# Load API keys
EXPERIMENT_ROOT = Path(__file__).resolve().parent
load_dotenv(EXPERIMENT_ROOT / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Tiny subset — first 5 prompts only
pairs = pd.read_json(EXPERIMENT_ROOT / "data" / "preprocessed_pairs.jsonl", lines=True).head(5)
print(f"Test: {len(pairs)} prompts × 1 cell × 2 rounds = {len(pairs) * 2} calls", flush=True)

OUTPUT_DIR = EXPERIMENT_ROOT / "data" / "test_run"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Run the anchor cell only with low parallelism (to keep the test cheap and obvious)
run_cell_batch_parallel(
    client=client,
    cell_id="C1F1D1",
    pairs=pairs,
    output_path=OUTPUT_DIR / "C1F1D1.jsonl",
    resume=True,
    max_workers=4,
)

print("\n✓ Test complete. Inspect data/test_run/C1F1D1.jsonl", flush=True)
