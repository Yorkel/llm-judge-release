"""Retry L1-swap items that parse-failed or were dropped.

Reads data/L1_swap/C1F1D1.jsonl, finds records where round1.verdict or
round2.verdict is None, plus any uids missing entirely. Re-runs only those
pairs and rewrites the JSONL with the retried records merged in. Backs up
the original JSONL with a timestamp suffix before overwriting.

Usage:
    cd llm-judge-release
    python retry_l1_failed.py
"""

import json
import os
import time
from pathlib import Path

import anthropic
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from run_l1_swap import ANCHOR_CELL, JUDGE_MODEL, run_judge_pair

EXPERIMENT_ROOT = Path(__file__).resolve().parent
JSONL_PATH = EXPERIMENT_ROOT / "data" / "L1_swap" / f"{ANCHOR_CELL}.jsonl"
PAIRS_PATH = EXPERIMENT_ROOT / "data" / "preprocessed_pairs.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def is_failed(rec):
    return (
        rec.get("round1", {}).get("verdict") is None
        or rec.get("round2", {}).get("verdict") is None
    )


def main():
    load_dotenv(EXPERIMENT_ROOT / ".env")
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    pairs = pd.read_json(PAIRS_PATH, lines=True).set_index("uid")
    existing = load_jsonl(JSONL_PATH) if JSONL_PATH.exists() else []
    by_uid = {r["uid"]: r for r in existing}

    failed_uids = [r["uid"] for r in existing if is_failed(r)]
    missing_uids = [u for u in pairs.index if u not in by_uid]
    to_retry = failed_uids + missing_uids

    print(f"Parse-failed: {len(failed_uids)}")
    print(f"Missing entirely: {len(missing_uids)}")
    print(f"Total to retry: {len(to_retry)}")
    if not to_retry:
        return

    backup = JSONL_PATH.with_suffix(f".jsonl.bak.{int(time.time())}")
    JSONL_PATH.rename(backup)
    print(f"Backed up original to {backup.name}")

    for uid in tqdm(to_retry, desc="retrying"):
        row = pairs.loc[uid]
        try:
            result = run_judge_pair(
                client, ANCHOR_CELL,
                question=row["prompt"],
                candidate_response=row["candidate_response"],
                baseline_response=row["baseline_response"],
            )
            by_uid[uid] = {
                "uid": uid,
                "spec_cell_id": ANCHOR_CELL,
                "round1": result["round1"],
                "round2": result["round2"],
                "model": JUDGE_MODEL,
                "timestamp": time.time(),
            }
        except Exception as e:
            print(f"Still failed on {uid}: {e}")

    with open(JSONL_PATH, "w", encoding="utf-8") as f:
        for rec in by_uid.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    still_failed = sum(1 for r in by_uid.values() if is_failed(r))
    print(f"Final: {len(by_uid)} records. Still failed: {still_failed}")


if __name__ == "__main__":
    main()
