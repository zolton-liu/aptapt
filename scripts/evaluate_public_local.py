#!/usr/bin/env python3
"""Run all Track-1 public units without requiring a local Docker daemon.

This is a development runner, not a replacement for the official container
harness.  It runs the agent against the unmodified public unit, stages a copy
of the grader-owned checks, rewrites container mount paths to their local
equivalents, and invokes pytest.  Results are committed after every unit so a
long run can be resumed safely.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluate_public import (
    card_metadata,
    load_json_if_present,
    load_tasks,
    run_process,
    trace_metrics,
    verifier_counts,
    write_aggregate,
)


def discover_tasks(official_repo: Path, task_list: Path | None) -> list[str]:
    if task_list is not None:
        return load_tasks(task_list.resolve())
    units = official_repo / "units"
    return sorted(path.parent.name for path in units.glob("*/card.toml"))


def _replace_container_paths(source: str, replacements: dict[str, str]) -> str:
    # Replace in one pass. Sequential str.replace would rewrite a newly inserted
    # host path again when it happens to end in a key such as ``/output``.
    keys = sorted(replacements, key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(key) for key in keys))
    return pattern.sub(lambda match: replacements[match.group(0)], source)


def stage_checker(unit: Path, run_dir: Path, output_dir: Path) -> Path:
    root = run_dir / "verifier_workspace"
    checks = root / "checks"
    shutil.copytree(unit / "checks", checks)
    shutil.copy2(unit / "card.toml", root / "task.toml")
    environment = root / "environment"
    environment.symlink_to(unit / "environment", target_is_directory=True)
    logs = run_dir / "verifier_logs"
    logs.mkdir()

    data = (unit / "environment" / "data").resolve()
    references = (checks / "reference_data").resolve()
    replacements = {
        "/tests/reference_data": str(references),
        "/logs/verifier": str(logs.resolve()),
        "/app/output": str(output_dir.resolve()),
        "/app/data": str(data),
        "/output": str(output_dir.resolve()),
        "/input": str(unit.resolve()),
        # Thin unit images copy the contents of environment/data into /app.
        "/app": str(data),
    }
    for path in checks.rglob("*.py"):
        original = path.read_text(encoding="utf-8")
        rewritten = _replace_container_paths(original, replacements)
        path.write_text(rewritten, encoding="utf-8")
    return checks / "test_outputs.py"


def classify_failure(
    *, agent_rc: int, verifier_rc: int, timed_out: bool, metrics: dict[str, Any]
) -> str:
    if timed_out:
        return "agent_timeout"
    if metrics.get("status") == "incomplete":
        return "agent_incomplete"
    if agent_rc != 0:
        return "agent_error"
    if verifier_rc != 0:
        return "verifier_failure"
    return "pass"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--task-list", type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("eval_runs"))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-endpoint", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="qwen2.5-coder:14b")
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--max-response-tokens", type=int, default=4000)
    parser.add_argument("--model-timeout-sec", type=float, default=360.0)
    parser.add_argument("--task-timeout-sec", type=float, default=900.0)
    parser.add_argument("--verifier-timeout-sec", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    official_repo = args.official_repo.resolve()
    tasks = discover_tasks(official_repo, args.task_list)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    if not tasks:
        raise SystemExit("no public units discovered")

    experiment_dir = (args.output_root / args.experiment).resolve()
    if experiment_dir.exists() and not args.resume:
        raise SystemExit(f"experiment already exists; use --resume: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "run_config.json").write_text(
        json.dumps(
            {
                "execution_mode": "local-public",
                "official_repo": str(official_repo),
                "n_tasks": len(tasks),
                "task_ids": tasks,
                "model_endpoint": args.model_endpoint,
                "model_name": args.model_name,
                "max_steps": args.max_steps,
                "max_response_tokens": args.max_response_tokens,
                "model_timeout_sec": args.model_timeout_sec,
                "task_timeout_sec": args.task_timeout_sec,
                "outer_deadline_forwarded": True,
                "framework_features": {name: os.environ.get(name, "1").lower() not in {"0", "false", "off"}
                                       for name in ("QFA_STAGE_PIPELINE", "QFA_REPAIR_CONTEXT", "QFA_VERSIONED_MEMORY")},
                "verifier_timeout_sec": args.verifier_timeout_sec,
                "reserve_sec": 5,
                "runner_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "starters_disabled": True,
                "context_memory_enabled": os.environ.get("QFA_CONTEXT_MEMORY", "1").lower()
                not in {"0", "false", "off"},
                "context_max_tokens": int(os.environ.get("QFA_CONTEXT_MAX_TOKENS", "16000")),
                "framework_source_sha256": hashlib.sha256(b"".join(
                    path.name.encode() + b"\0" + path.read_bytes()
                    for path in sorted((project_root / "src/qfa_agent").glob("*.py"))
                )).hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    results_path = experiment_dir / "results.jsonl"
    rows = []
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    completed = {str(row["task_id"]) for row in rows}

    for position, task_id in enumerate(tasks, start=1):
        if task_id in completed:
            print(f"[{position}/{len(tasks)}] skip {task_id} (already recorded)", flush=True)
            continue
        unit = official_repo / "units" / task_id
        if not (unit / "card.toml").exists():
            raise SystemExit(f"missing unit: {unit}")
        run_dir = experiment_dir / task_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        output_dir = run_dir / "output"
        scratch_dir = run_dir / "scratch"
        output_dir.mkdir(parents=True)
        scratch_dir.mkdir()
        print(f"[{position}/{len(tasks)}] run {task_id}", flush=True)

        env = os.environ.copy()
        env.update(
            {
                "PYTHONPATH": str(project_root / "src"),
                "MODEL_ENDPOINT": args.model_endpoint,
                "MODEL_NAME": args.model_name,
                "MODEL_TOKEN": "local-development-token",
                "QFA_DISABLE_STARTERS": "1",
                "QFA_JSON_MODE": "1",
                "QFA_MAX_STEPS": str(args.max_steps),
                "QFA_MAX_RESPONSE_TOKENS": str(args.max_response_tokens),
                "QFA_MODEL_TIMEOUT_SEC": str(args.model_timeout_sec),
                "QFA_RESERVE_SEC": "5",
                "QFA_TRACE_PATH": str(run_dir / "trajectory.jsonl"),
                "QFA_RUN_METRICS_PATH": str(run_dir / "run_metrics.json"),
                "QFA_SCRATCH_DIR": str(scratch_dir),
            }
        )
        command = [
            args.python,
            "-m",
            "qfa_agent.cli",
            "solve",
            "--task-dir",
            str(unit),
            "--out",
            str(output_dir),
            "--time-budget-sec",
            str(args.task_timeout_sec),
        ]
        started = time.monotonic()
        try:
            completed_process = subprocess.run(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=args.task_timeout_sec,
                check=False,
            )
            agent_rc = completed_process.returncode
            agent_stdout = completed_process.stdout
            agent_stderr = completed_process.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            agent_rc = 124
            agent_stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            agent_stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            timed_out = True
        wall_time = time.monotonic() - started
        (run_dir / "agent.stdout.log").write_text(agent_stdout, encoding="utf-8")
        (run_dir / "agent.stderr.log").write_text(agent_stderr, encoding="utf-8")

        verifier_rc = 125
        verifier_stdout = ""
        verifier_stderr = ""
        if (unit / "checks" / "test_outputs.py").exists():
            checker = stage_checker(unit, run_dir, output_dir)
            verifier_env = os.environ.copy()
            verifier_env.update(
                {
                    "OUTPUT_DIR": str(output_dir),
                    "DATA_DIR": str(unit / "environment" / "data"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
            )
            verifier_rc, verifier_stdout, verifier_stderr, _, _ = run_process(
                [
                    args.python,
                    "-m",
                    "pytest",
                    "-q",
                    "-o",
                    "addopts=",
                    "-p",
                    "no:cacheprovider",
                    str(checker),
                ],
                timeout=args.verifier_timeout_sec,
            )
        (run_dir / "verifier.stdout.log").write_text(verifier_stdout, encoding="utf-8")
        (run_dir / "verifier.stderr.log").write_text(verifier_stderr, encoding="utf-8")

        metrics = load_json_if_present(run_dir / "run_metrics.json")
        traced = trace_metrics(run_dir / "trajectory.jsonl")
        verifier_text = verifier_stdout + "\n" + verifier_stderr
        verifier_passed, verifier_failed = verifier_counts(verifier_text)
        error_match = re.search(r"(\d+) errors?\b", verifier_text)
        verifier_errors = int(error_match.group(1)) if error_match else 0
        reward = int(agent_rc == 0 and verifier_rc == 0 and not timed_out)
        failure_type = classify_failure(
            agent_rc=agent_rc,
            verifier_rc=verifier_rc,
            timed_out=timed_out,
            metrics=metrics,
        )
        metadata = card_metadata(unit)
        row = {
            "experiment": args.experiment,
            "task_id": task_id,
            **metadata,
            "reward": reward,
            "failure_type": failure_type,
            "agent_exit_code": agent_rc,
            "verifier_exit_code": verifier_rc,
            "verifier_passed": verifier_passed,
            "verifier_failed": verifier_failed,
            "verifier_errors": verifier_errors,
            "duration_sec": float(metrics.get("duration_sec", wall_time)),
            "steps": int(metrics.get("steps", traced["steps"]) or traced["steps"]),
            "model_calls": int(metrics.get("model_calls", traced["model_calls"]) or traced["model_calls"]),
            "input_tokens": int(metrics.get("input_tokens", traced["input_tokens"]) or traced["input_tokens"]),
            "output_tokens": int(metrics.get("output_tokens", traced["output_tokens"]) or traced["output_tokens"]),
            "starter_used": bool(metrics.get("starter_used", False)),
            "routed_category": metrics.get("routed_category", ""),
            "solver_architecture": metrics.get("solver_architecture", ""),
            "model_name": metrics.get("model_name", args.model_name),
            "execution_mode": "local-public",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        write_aggregate(experiment_dir, rows)
        print(
            f"  reward={reward} failure={failure_type} tests={verifier_passed}/{verifier_failed} "
            f"calls={row['model_calls']} tokens={row['input_tokens']}+{row['output_tokens']} "
            f"duration={row['duration_sec']:.1f}s",
            flush=True,
        )

    write_aggregate(experiment_dir, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
