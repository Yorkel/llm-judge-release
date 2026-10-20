"""Validate frozen input hashes and October record integrity without API access."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-4.1-2025-04-14"
LABELS = {"A>>B", "A>B", "A=B", "B>A", "B>>A"}


def records(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(root=ROOT):
    root = Path(root).resolve()
    with (root / "data/manifest.csv").open(newline="", encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    listed = {r["path"] for r in manifest}
    actual = {str(p.relative_to(root)) for p in (root / "data").rglob("*.jsonl")}
    if len(listed) != len(manifest) or listed != actual:
        raise ValueError("Frozen input manifest and data/*.jsonl inventory differ")
    for item in manifest:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root / "data"):
            raise ValueError(f"Input escapes data directory: {item['path']}")
        blob = path.read_bytes()
        if len(blob) != int(item["bytes"]) or hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen input changed: {item['path']}")
        if len(blob.splitlines()) != int(item["rows"]):
            raise ValueError(f"Input row count differs: {item['path']}")

    pairs = records(root / "data/preprocessed_pairs.jsonl")
    expected_ids = {r["uid"] for r in pairs}
    if len(pairs) != 500 or len(expected_ids) != 500:
        raise ValueError("Expected 500 unique frozen prompt/response pairs")
    cells = [f"C{c}F{f}D{d}" for c in (1, 2) for f in (1, 2) for d in (1, 2)]
    expected_files = {f"data/october_rerun/{run}/{cell}.jsonl"
                      for run in ("run1", "run2") for cell in cells}
    expected_files.add("data/october_rerun/run3/C1F1D1.jsonl")
    observed_files = {name for name in listed if name.startswith("data/october_rerun/")}
    if observed_files != expected_files:
        raise ValueError("October data must contain eight setups twice and a third default run")

    failures = 0
    fingerprints = Counter()
    dates = set()
    complete = set(expected_ids)
    total_rounds = 0
    for name in sorted(expected_files):
        data = records(root / name)
        ids = [r["uid"] for r in data]
        if len(ids) != 500 or len(set(ids)) != 500 or set(ids) != expected_ids:
            raise ValueError(f"Duplicate, missing or unexpected prompt ID: {name}")
        file_complete = set()
        for row in data:
            if row["spec_cell_id"] != Path(name).stem or row["model"] != MODEL:
                raise ValueError(f"Unexpected condition/model metadata: {name}")
            dates.add(datetime.fromtimestamp(row["timestamp"], timezone.utc).date().isoformat())
            valid = True
            for key in ("round1", "round2"):
                response = row[key]
                total_rounds += 1
                if response.get("served_model") != MODEL:
                    raise ValueError(f"Unexpected served model: {name}")
                fingerprints[response.get("system_fingerprint")] += 1
                raw = response["raw"].upper()
                matches = re.findall(r"\[\[([AB<>=]+)\]\]", raw) or re.findall(r"\[([AB<>=]+)\]", raw)
                parsed = matches[-1] if matches else None
                if response["verdict"] != parsed or response["parse_success"] != (parsed is not None):
                    raise ValueError(f"Stored verdict/parse flag differs from raw response: {name}")
                if parsed not in LABELS:
                    failures += 1
                    valid = False
                    if parsed is not None or response["output_tokens"] != 16000:
                        raise ValueError(f"Unexpected invalid verdict: {name}")
            if valid:
                file_complete.add(row["uid"])
        complete.intersection_update(file_complete)
    if (total_rounds, failures, len(complete)) != (17000, 4, 496):
        raise ValueError("October round count, failures or complete sample differs from release")
    summary = {"frozen_files": len(manifest), "october_files": len(expected_files),
               "october_records": total_rounds // 2, "rounds": total_rounds,
               "unparsed_rounds": failures, "complete_across_all_runs": len(complete),
               "served_model": MODEL, "collection_dates_utc": sorted(dates),
               "distinct_system_fingerprints": len(fingerprints)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Release root to validate")
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root), indent=2))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Release validation failed: {error}\n")


if __name__ == "__main__":
    main()
