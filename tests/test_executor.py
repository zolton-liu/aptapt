from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from qfa_agent.executor import run_bounded


class ExecutorTests(unittest.TestCase):
    def test_captures_and_truncates_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_bounded(
                [sys.executable, "-c", "print('x' * 20000)"],
                cwd=Path(temporary),
                timeout_sec=5,
                output_limit_bytes=1000,
            )
        self.assertTrue(result.passed)
        self.assertGreater(result.truncated_bytes, 0)
        self.assertIn("omitted", result.output)

    def test_times_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = run_bounded(
                [sys.executable, "-c", "import time; time.sleep(5)"],
                cwd=Path(temporary),
                timeout_sec=0.2,
            )
        self.assertTrue(result.timed_out)
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()

