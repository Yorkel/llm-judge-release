"""Rebuild and verify the frozen October analysis, offline, without changing matching files.

Run from the release root (or use an absolute path to this script from elsewhere):
    python analysis/build_october.py
    python analysis/build_october.py --check-only
    python analysis/build_october.py --output-dir /path/to/verified-tables

All analyses run sequentially in a temporary directory, using their original formulas,
20,000 bootstrap draws and seeds. The build checks data/manifest.csv first, then compares
all 13 generated tables with the shipped results. It fails before installing anything
if an input, reference, schema, or result differs. Matching destination files retain their
original bytes. An output directory receives only verified missing tables. This command
does not import the API runners or make API calls. Figures have a separate build command.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results" / "october_rerun"
CELLS = ("C1F1D1", "C1F1D2", "C1F2D1", "C1F2D2", "C2F1D1", "C2F1D2", "C2F2D1", "C2F2D2")
JUDGMENTS = tuple(
    f"data/october_rerun/{run}/{cell}.jsonl"
    for run in ("run1", "run2", "run3")
    for cell in (CELLS if run != "run3" else ("C1F1D1",))
)
STEPS = (
    ("october_step2_table1", ("step2_table1_october.csv",)),
    ("october_step3_kappa", ("step3_kappa_october.csv",)),
    ("october_step3b_kappa_between", ("step3b_kappa_between_october.csv",)),
    ("october_step4_overlap", ("step4_overlap_october.csv",)),
    ("october_step5_breakdowns", ("step5_breakdowns_october.csv",)),
    ("october_step8_reversals", ("step8_reversals_october.csv",)),
    ("october_step9_half_point_as_tie", ("step9_half_point_as_tie_october.csv",)),
    ("october_step10_closeness", ("step10_closeness_october.csv",)),
    ("october_step11_scope_and_half_point", ("step11_scope_and_half_point_october.csv",)),
    ("october_step12_scores", ("step12_conditions_october.csv", "step12_factorial_october.csv", "step12_tally_october.csv")),
    ("october_step13_thresholds", ("step13_thresholds_october.csv",)),
)
OUTPUTS = tuple(name for _, names in STEPS for name in names)
TOLERANCE = 1e-10


def verify_manifest(root: Path, manifest: Path) -> None:
    """Verify every listed frozen JSONL; require coverage of all October inputs."""
    required = {"data/preprocessed_pairs.jsonl", *JUDGMENTS}
    seen = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"path", "sha256", "bytes", "rows"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest is missing required columns: {manifest}")
        for entry in reader:
            name = entry["path"]
            if name in seen:
                raise ValueError(f"Duplicate manifest path: {name}")
            seen.add(name)
            path = (root / name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(root.resolve()):
                raise ValueError(f"Manifest path is outside the release: {name}")
            digest = hashlib.sha256()
            size = rows = 0
            with path.open("rb") as raw:
                for line in raw:
                    digest.update(line)
                    size += len(line)
                    rows += bool(line.strip())
            actual = digest.hexdigest(), size, rows
            expected = entry["sha256"], int(entry["bytes"]), int(entry["rows"])
            if actual != expected:
                raise ValueError(f"Frozen input differs from manifest: {name}")
    if required - seen:
        raise ValueError(f"Manifest omits required inputs: {sorted(required - seen)}")
    print(f"Verified hashes, byte counts and row counts for {len(seen)} frozen inputs.", flush=True)


def verify_october_records(root: Path) -> None:
    """Reject duplicate/missing IDs and malformed stored judgments before any joins."""
    with (root / "data/preprocessed_pairs.jsonl").open(encoding="utf-8") as handle:
        ids = [json.loads(line)["uid"] for line in handle if line.strip()]
    expected = set(ids)
    if len(ids) != 500 or len(expected) != 500:
        raise ValueError("Expected exactly 500 unique frozen input prompt IDs")
    valid_verdicts = {None, "A>>B", "A>B", "A=B", "B>A", "B>>A"}
    for name in JUDGMENTS:
        path = root / name
        seen = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                record = json.loads(line)
                uid = record["uid"]
                if uid in seen or uid not in expected:
                    raise ValueError(f"Duplicate or unknown prompt ID: {name}:{line_number}")
                seen.add(uid)
                if record["spec_cell_id"] != path.stem or record["model"] != "gpt-4.1-2025-04-14":
                    raise ValueError(f"Unexpected cell or requested model: {name}:{line_number}")
                for round_name in ("round1", "round2"):
                    judgment = record[round_name]
                    verdict = judgment["verdict"]
                    if (verdict not in valid_verdicts
                            or judgment["parse_success"] != (verdict is not None)
                            or judgment["served_model"] != "gpt-4.1-2025-04-14"):
                        raise ValueError(f"Malformed judgment metadata: {name}:{line_number}:{round_name}")
        if seen != expected:
            raise ValueError(f"Missing prompt IDs: {name}")
    print("Verified 17 judgment files, 8,500 unique-per-file records and pinned served models.", flush=True)


def compare_table(expected: Path, actual: Path) -> None:
    """Require identical rows/schema/strings and numerical equality within roundoff."""
    if not expected.is_file():
        raise ValueError(f"Missing reference table: {expected}")
    if not actual.is_file():
        raise ValueError(f"Missing generated table: {actual}")
    try:
        pd.testing.assert_frame_equal(
            pd.read_csv(expected), pd.read_csv(actual),
            check_exact=False, rtol=TOLERANCE, atol=TOLERANCE,
        )
    except AssertionError as error:
        raise ValueError(f"Table mismatch for {expected.name}:\n{error}") from error


def build_tables(staged: Path) -> None:
    for module_name, names in STEPS:
        module = importlib.import_module(module_name)
        original_out = module.OUT
        original_self = getattr(module, "SELF", None)
        module.OUT = staged if len(names) > 1 else staged / names[0]
        if original_self is not None:
            module.SELF = staged / "step3_kappa_october.csv"
        started = time.monotonic()
        print(f"Running {module_name} ...", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                module.main()
        finally:
            module.OUT = original_out
            if original_self is not None:
                module.SELF = original_self
        print(f"  generated {len(names)} table(s) in {time.monotonic() - started:.1f}s", flush=True)


def verify_and_install(staged: Path, reference: Path, destination: Path | None) -> None:
    """Check every result/destination before writing; retain all matching bytes."""
    for name in OUTPUTS:
        compare_table(reference / name, staged / name)
    missing = []
    if destination is not None:
        for name in OUTPUTS:
            target = destination / name
            if target.exists():
                compare_table(target, staged / name)
            else:
                missing.append(name)
        if missing:
            destination.mkdir(parents=True, exist_ok=True)
            for name in missing:
                # A same-directory temporary file makes each installation atomic.
                with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                    pending = Path(handle.name)
                try:
                    shutil.copyfile(staged / name, pending)
                    os.replace(pending, destination / name)
                finally:
                    pending.unlink(missing_ok=True)
    print(f"Verified all {len(OUTPUTS)} tables against shipped references (tolerance {TOLERANCE:g}).", flush=True)
    if destination is not None:
        print(f"Installed {len(missing)} missing tables; matching destination files retain their bytes.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check-only", action="store_true", help="Rebuild and verify without installing any outputs")
    parser.add_argument("--output-dir", type=Path, help="Install verified missing tables in a separate directory")
    args = parser.parse_args()
    if args.check_only and args.output_dir is not None:
        parser.error("--check-only and --output-dir cannot be combined")
    try:
        for name in OUTPUTS:
            if not (REFERENCE / name).is_file():
                raise ValueError(f"Missing reference table: {REFERENCE / name}")
        verify_manifest(ROOT, ROOT / "data" / "manifest.csv")
        verify_october_records(ROOT)
        destination = None if args.check_only else (args.output_dir or REFERENCE)
        with tempfile.TemporaryDirectory(prefix="llm-judge-october-") as scratch:
            staged = Path(scratch)
            build_tables(staged)
            verify_and_install(staged, REFERENCE, destination)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"October build failed: {error}\n")


if __name__ == "__main__":
    main()
