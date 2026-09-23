#!/usr/bin/env python3
"""Replace invalid evaluation records with reruns and rebuild summary files."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_rows(args.base / "results.jsonl")
    rerun_rows = read_rows(args.rerun / "results.jsonl")
    replacement = {str(row["task_id"]): row for row in rerun_rows}
    base_ids = {str(row["task_id"]) for row in base_rows}
    unknown = sorted(set(replacement) - base_ids)
    if unknown:
        raise SystemExit(f"rerun contains tasks absent from base: {unknown}")
    if len(replacement) != len(rerun_rows):
        raise SystemExit("rerun contains duplicate task ids")

    experiment = args.out.name
    merged: list[dict[str, object]] = []
    replaced: list[str] = []
    for original in base_rows:
        task_id = str(original["task_id"])
        row = dict(replacement.get(task_id, original))
        if task_id in replacement:
            replaced.append(task_id)
        row["experiment"] = experiment
        merged.append(row)
    if len(merged) != len(base_rows):
        raise SystemExit("merged row count changed")

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in merged),
        encoding="utf-8",
    )
    fieldnames = list(merged[0])
    with (args.out / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)

    summary = {
        "experiment": experiment,
        "n_tasks": len(merged),
        "n_passed": sum(int(row.get("reward", 0)) for row in merged),
        "pass_at_1": sum(int(row.get("reward", 0)) for row in merged) / len(merged),
        "mean_duration_sec": sum(float(row.get("duration_sec", 0)) for row in merged) / len(merged),
        "total_input_tokens": sum(int(row.get("input_tokens", 0)) for row in merged),
        "total_output_tokens": sum(int(row.get("output_tokens", 0)) for row in merged),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    provenance = {
        "base": str(args.base.resolve()),
        "rerun": str(args.rerun.resolve()),
        "replaced_task_ids": replaced,
    }
    (args.out / "merge_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
