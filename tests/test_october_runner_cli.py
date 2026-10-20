"""Offline checks: no API dependency, credentials, client or calls are needed."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_october_rerun.py"


class RunnerCliTests(unittest.TestCase):
    def run_cli(self, *args):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("PYTHONPATH", None)
        # -S disables site-packages: reaching pandas/openai/dotenv would fail.
        return subprocess.run(
            [sys.executable, "-S", str(SCRIPT), *args],
            cwd=tempfile.gettempdir(), env=env, capture_output=True, text=True,
        )

    def assert_rejected_before_client(self, result, message):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("OPENAI_API_KEY", result.stderr)

    def test_help_without_credentials_or_api_dependencies(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-root", result.stdout)
        self.assertIn("new_api_runs", result.stdout)

    def test_import_without_credentials_or_api_dependencies(self):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import run_october_rerun"],
            cwd=ROOT, env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_run_names_reject_paths_and_traversal(self):
        for name in ("../run1", "/tmp/run1", "run1/child", "run1\\child", ".", ".."):
            with self.subTest(name=name):
                self.assert_rejected_before_client(
                    self.run_cli("--runs", name), "run names must",
                )

    def test_frozen_data_output_is_rejected(self):
        for destination in (ROOT / "data", ROOT / "data" / "new_run"):
            with self.subTest(destination=destination):
                self.assert_rejected_before_client(
                    self.run_cli("--output-root", str(destination)), "frozen data/",
                )

    def test_symlink_into_frozen_data_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            alias = root / "data_alias"
            alias.symlink_to(ROOT / "data", target_is_directory=True)
            self.assert_rejected_before_client(
                self.run_cli("--output-root", str(alias)), "frozen data/",
            )
            (root / "run1").symlink_to(ROOT / "data", target_is_directory=True)
            self.assert_rejected_before_client(
                self.run_cli("--output-root", str(root), "--runs", "run1"),
                "frozen data/",
            )


if __name__ == "__main__":
    unittest.main()
