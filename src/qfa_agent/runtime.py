"""Compatibility runtime for model-authored Python programs."""

from __future__ import annotations

import os
import csv
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import kurtosis, norm, skew


_STANDARD_JSON_DEFAULT = json.JSONEncoder.default


def _finance_json_default(self, value):
    """Serialize common numeric-library scalars without permitting NaN."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return _STANDARD_JSON_DEFAULT(self, value)


class TaskPath:
    """Route legacy task-relative output paths without hiding real input paths."""

    def __init__(self, input_root: Path, output_root: Path):
        self.input_root = input_root
        self.output_root = output_root

    def __truediv__(self, value: object) -> Path:
        relative = os.fspath(value).replace("\\", "/").lstrip("/")
        if relative == "output":
            return self.output_root
        if relative.startswith("output/"):
            return self.output_root / relative.removeprefix("output/")
        return self.input_root / relative

    def __fspath__(self) -> str:
        return os.fspath(self.input_root)

    def __str__(self) -> str:
        return str(self.input_root)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw:
        print("runtime requires a Python script path", file=sys.stderr)
        return 2
    script = Path(raw[0]).resolve(strict=True)
    input_root = Path(os.environ["TASK_DIR"]).resolve(strict=True)
    output_root = Path(os.environ["OUTPUT_DIR"]).resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)
    json.JSONEncoder.default = _finance_json_default
    sys.argv = [str(script), *raw[1:]]
    runpy.run_path(
        str(script),
        run_name="__main__",
        init_globals={
            "task_dir": TaskPath(input_root, output_root),
            "output_dir": output_root,
            "Path": Path,
            "os": os,
            "csv": csv,
            "json": json,
            "math": math,
            # Compatibility names for common small-model omissions.  The
            # generated program remains visible and auditable; these globals
            # only prevent a correct numerical draft from burning several
            # House calls adding conventional imports one at a time.
            "np": np,
            "pd": pd,
            "scipy": scipy,
            "norm": norm,
            "skew": skew,
            "kurtosis": kurtosis,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
