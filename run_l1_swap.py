"""Standalone runner for the L1 vendor swap (anchor cell only, Claude Sonnet judge).

Mirrors run_full_factorial.py but uses Anthropic SDK for the L1 variant judge.
Runs only the anchor cell C1F1D1 — same prompts, same candidates, same baseline
— with claude-sonnet-4-5-20250929 instead of gpt-4.1.

Usage:
    cd llm-judge-release
    nohup python run_l1_swap.py > run_l1_swap.log 2>&1 &
    tail -f run_l1_swap.log

Scope: 500 prompts × 1 cell × 2 rounds = 1,000 calls. Estimated cost ~$22 on
claude-sonnet-4-5-20250929. Estimated runtime ~30-60 min with max_workers=4.

Resumes via skip-already-judged in JSONL, same pattern as the OpenAI runner.
"""

import concurrent.futures
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from _prompts import get_system_prompt, get_user_prompt

# Locked L1 swap config
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 0.0
MAX_TOKENS = 4096
ANCHOR_CELL = "C1F1D1"

# Same regex patterns as _pipeline.py (matches v2.0 default)
PRIMARY_REGEX = re.compile(r"\[\[([AB<>=]+)\]\]")
FALLBACK_REGEX = re.compile(r"\[([AB<>=]+)\]")


def parse_verdict(text: str):
    """Extract verdict tag — primary then fallback regex."""
    upper = text.upper()
    matches = [m for m in PRIMARY_REGEX.findall(upper) if m]
    if matches:
        return matches[-1].strip("\n")
    matches = [m for m in FALLBACK_REGEX.findall(upper) if m]
    if matches:
        return matches[-1].strip("\n")
    return None


def call_claude_judge(client, system_prompt, user_prompt, max_retries=8):
    """Single Claude judge call with retry-on-rate-limit + prompt caching.

    Caches the system prompt block (stable across all calls) via cache_control.
    """
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[{"role": "user", "content": user_prompt}],
            )
            break
        except Exception as e:
            err = str(e)
            is_rate = "429" in err or "rate_limit" in err.lower() or "overloaded" in err.lower()
            if is_rate and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise

    raw = resp.content[0].text
    verdict = parse_verdict(raw)
    usage = resp.usage
    return {
        "raw": raw,
        "verdict": verdict,
        "parse_success": verdict is not None,
        "input_tokens": usage.input_tokens,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
        "output_tokens": usage.output_tokens,
    }


def run_judge_pair(client, cell_id, question, candidate_response, baseline_response):
    """Two-round bidirectional swap (Arena-Hard default)."""
    system_prompt = get_system_prompt(cell_id)
    # Round 1: A=baseline, B=candidate
    user_r1 = get_user_prompt(cell_id, question, baseline_response, candidate_response)
    r1 = call_claude_judge(client, system_prompt, user_r1)
    # Round 2: positions swapped
    user_r2 = get_user_prompt(cell_id, question, candidate_response, baseline_response)
    r2 = call_claude_judge(client, system_prompt, user_r2)
    return {"round1": r1, "round2": r2}


def run_cell_batch(client, cell_id, pairs, output_path, max_workers=4):
    """Resumable parallel runner — skip uids already in output JSONL."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done_uids = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                done_uids.add(json.loads(line)["uid"])
        if done_uids:
            print(f"[{cell_id}] resuming — {len(done_uids)} uids already done", flush=True)

    to_run = pairs[~pairs["uid"].isin(done_uids)]
    if len(to_run) == 0:
        print(f"[{cell_id}] already complete", flush=True)
        return

    rows = to_run.to_dict("records")

    def process_one(row):
        try:
            result = run_judge_pair(
                client,
                cell_id,
                question=row["prompt"],
                candidate_response=row["candidate_response"],
                baseline_response=row["baseline_response"],
            )
            return {
                "uid": row["uid"],
                "spec_cell_id": cell_id,
                "round1": result["round1"],
                "round2": result["round2"],
                "model": JUDGE_MODEL,
                "timestamp": time.time(),
            }
        except Exception as e:
            print(f"[{cell_id}] error on uid={row['uid']}: {e}", flush=True)
            return None

    with open(output_path, "a", encoding="utf-8") as f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_one, row) for row in rows]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{cell_id} (Claude, parallel x{max_workers})",
            ):
                record = future.result()
                if record is not None:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()


# === Main ===
if __name__ == "__main__":
    EXPERIMENT_ROOT = Path(__file__).resolve().parent
    load_dotenv(EXPERIMENT_ROOT / ".env")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not found in .env")
    client = anthropic.Anthropic(api_key=api_key)

    pairs = pd.read_json(EXPERIMENT_ROOT / "data" / "preprocessed_pairs.jsonl", lines=True)
    print(f"Loaded {len(pairs)} pairs", flush=True)

    OUTPUT_DIR = EXPERIMENT_ROOT / "data" / "L1_swap"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{ANCHOR_CELL}.jsonl"

    print(f"\n{'='*60}", flush=True)
    print(f"L1 SWAP — start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Anchor cell {ANCHOR_CELL} × {len(pairs)} prompts × 2 rounds = {len(pairs) * 2} calls", flush=True)
    print(f"Judge model: {JUDGE_MODEL}", flush=True)
    print(f"{'='*60}\n", flush=True)

    start = time.time()
    run_cell_batch(client, ANCHOR_CELL, pairs, output_path, max_workers=4)
    elapsed = timedelta(seconds=int(time.time() - start))
    print(f"\n✓ L1 swap complete | total time {elapsed}", flush=True)
