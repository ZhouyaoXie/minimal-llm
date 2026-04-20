"""Smoke test: train CLI loads config and exits successfully."""

import subprocess
import sys
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


class TestTrainSmoke(unittest.TestCase):
    def test_train_cli_runs_with_config(self) -> None:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "train.train",
                "-s",
                "42",
                "-c",
                "configs/train.yaml",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            proc.returncode,
            0,
            msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}",
        )
        self.assertIn("seed: 42", proc.stdout)
        self.assertIn("configs/train.yaml", proc.stdout)
        self.assertIn("steps", proc.stdout)


if __name__ == "__main__":
    unittest.main()
