"""Small failure-path checks; the full numerical rebuild is build_october.py."""
import contextlib
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_october as build


class OctoberBuildTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.reference = self.root / "reference"
        self.staged = self.root / "staged"
        self.reference.mkdir()
        self.staged.mkdir()
        for name in ("a.csv", "b.csv"):
            (self.reference / name).write_text("label,value\nexample,0.5000000000000000\n")
            (self.staged / name).write_text("label,value\nexample,0.5\n")
        self.outputs = patch.object(build, "OUTPUTS", ("a.csv", "b.csv"))
        self.outputs.start()
        self.addCleanup(self.outputs.stop)

    def install(self, destination):
        with contextlib.redirect_stdout(io.StringIO()):
            build.verify_and_install(self.staged, self.reference, destination)

    def test_matching_files_keep_original_bytes(self):
        before = {p.name: p.read_bytes() for p in self.reference.iterdir()}
        self.install(self.reference)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.reference.iterdir()})

    def test_verified_missing_tables_are_installed(self):
        destination = self.root / "export"
        self.install(destination)
        self.assertEqual((destination / "a.csv").read_bytes(), (self.staged / "a.csv").read_bytes())
        self.assertEqual((destination / "b.csv").read_bytes(), (self.staged / "b.csv").read_bytes())

    def test_late_result_mismatch_prevents_all_installation(self):
        (self.staged / "b.csv").write_text("label,value\nexample,0.6\n")
        destination = self.root / "export"
        with self.assertRaisesRegex(ValueError, "Table mismatch"):
            self.install(destination)
        self.assertFalse(destination.exists())

    def test_missing_reference_is_not_replaced_with_generated_truth(self):
        (self.reference / "b.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing reference"):
            self.install(self.reference)
        self.assertFalse((self.reference / "b.csv").exists())

    def test_schema_mismatch_fails(self):
        (self.staged / "b.csv").write_text("label,renamed\nexample,0.5\n")
        with self.assertRaisesRegex(ValueError, "Table mismatch"):
            self.install(None)

    def test_manifest_detects_corruption(self):
        data = self.root / "data"
        data.mkdir()
        source = data / "preprocessed_pairs.jsonl"
        source.write_bytes(b'{"uid":"one"}\n')
        raw = source.read_bytes()
        manifest = data / "manifest.csv"
        with manifest.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("path", "sha256", "bytes", "rows"))
            writer.writerow(("data/preprocessed_pairs.jsonl", hashlib.sha256(raw).hexdigest(), len(raw), 1))
        with patch.object(build, "JUDGMENTS", ()), contextlib.redirect_stdout(io.StringIO()):
            build.verify_manifest(self.root, manifest)
            source.write_bytes(b'{"uid":"two"}\n')
            with self.assertRaisesRegex(ValueError, "differs from manifest"):
                build.verify_manifest(self.root, manifest)


if __name__ == "__main__":
    unittest.main()
