#!/usr/bin/env python3
"""Run a reproducible, starter-disabled T1 public evaluation slice."""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_tasks(path: Path) -> list[str]:
    tasks = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            tasks.append(value)
    if not tasks:
        raise ValueError(f"task list is empty: {path}")
    if len(tasks) != len(set(tasks)):
        raise ValueError("task list contains duplicates")
    return tasks


def card_metadata(unit: Path) -> dict[str, Any]:
    with (unit / "card.toml").open("rb") as handle:
        card = tomllib.load(handle)
    return {
        "category": card.get("metadata", {}).get("category", "unknown"),
        "difficulty": card.get("metadata", {}).get("difficulty", "unknown"),
        "declared_timeout_sec": float(card.get("agent", {}).get("timeout_sec", 1800)),
    }


def safe_container_name(experiment: str, task: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"qfa-{experiment}-{task}")
    return value[:120].strip("-.")


def run_process(command: list[str], *, timeout: float) -> tuple[int, str, str, float, bool]:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        return (
            completed.returncode,
            completed.stdout,
            completed.stderr,
            time.monotonic() - started,
            False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, stdout, stderr, time.monotonic() - started, True


def docker_remove(container_name: str) -> None:
    subprocess.run(
        ["docker", "rm", "-f", container_name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def load_json_if_present(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def trace_metrics(path: Path) -> dict[str, int]:
    result = {"model_calls": 0, "input_tokens": 0, "output_tokens": 0, "steps": 0}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        result["steps"] = max(result["steps"], int(event.get("step", 0) or 0))
        if event.get("event") == "model_call":
            result["model_calls"] += 1
            result["input_tokens"] += int(event.get("input_tokens", 0) or 0)
            result["output_tokens"] += int(event.get("output_tokens", 0) or 0)
    return result


def verifier_counts(text: str) -> tuple[int, int]:
    passed = re.search(r"(\d+) passed", text)
    failed = re.search(r"(\d+) failed", text)
    return int(passed.group(1)) if passed else 0, int(failed.group(1)) if failed else 0


def write_aggregate(experiment_dir: Path, rows: list[dict[str, Any]]) -> None:
    jsonl = experiment_dir / "results.jsonl"
    jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    fields = [
        "experiment", "task_id", "category", "difficulty", "reward", "failure_type",
        "agent_exit_code", "verifier_exit_code", "verifier_passed", "verifier_failed",
        "verifier_errors",
        "duration_sec", "steps", "model_calls", "input_tokens", "output_tokens",
        "starter_used", "routed_category", "solver_architecture", "model_name",
    ]
    with (experiment_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(int(row["reward"]) for row in rows)
    summary = {
        "experiment": experiment_dir.name,
        "n_tasks": len(rows),
        "n_passed": passed,
        "pass_at_1": passed / len(rows) if rows else 0.0,
        "total_input_tokens": sum(int(row.get("input_tokens", 0) or 0) for row in rows),
        "total_output_tokens": sum(int(row.get("output_tokens", 0) or 0) for row in rows),
        "mean_duration_sec": (
            sum(float(row.get("duration_sec", 0.0) or 0.0) for row in rows) / len(rows)
            if rows else 0.0
        ),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    (experiment_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--task-list", type=Path, default=Path("configs/baseline_tasks.txt"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("eval_runs"))
    parser.add_argument("--agent-image", default="agenthon-t1-minimal:dev")
    parser.add_argument("--verifier-image", default="finance-bench-sandbox:latest")
    parser.add_argument("--model-endpoint", default="http://host.docker.internal:11434/v1")
    parser.add_argument("--model-name", default="qwen2.5:7b")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--max-response-tokens", type=int, default=4000)
    parser.add_argument("--model-timeout-sec", type=float, default=180.0)
    parser.add_argument("--task-timeout-sec", type=float, default=420.0)
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    official_repo = args.official_repo.resolve()
    tasks = load_tasks(args.task_list.resolve())
    if args.limit is not None:
        tasks = tasks[: args.limit]
    experiment_dir = (args.output_root / args.experiment).resolve()
    if experiment_dir.exists() and not args.resume:
        raise SystemExit(f"experiment already exists; use --resume: {experiment_dir}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    existing_path = experiment_dir / "results.jsonl"
    rows = []
    if existing_path.exists():
        rows = [json.loads(line) for line in existing_path.read_text().splitlines() if line.strip()]
    completed_tasks = {row["task_id"] for row in rows}

    for position, task_id in enumerate(tasks, start=1):
        if task_id in completed_tasks:
            print(f"[{position}/{len(tasks)}] skip {task_id} (already recorded)", flush=True)
            continue
        unit = official_repo / "units" / task_id
        if not (unit / "card.toml").exists():
            raise SystemExit(f"unit is not materialized in sparse checkout: {unit}")
        metadata = card_metadata(unit)
        run_dir = experiment_dir / task_id
        run_dir.mkdir(parents=True, exist_ok=False)
        output_dir = run_dir / "output"
        output_dir.mkdir()
        container_name = safe_container_name(args.experiment, task_id)
        print(f"[{position}/{len(tasks)}] run {task_id}", flush=True)

        agent_environment = [
            "-e", f"MODEL_ENDPOINT={args.model_endpoint}",
            "-e", f"MODEL_NAME={args.model_name}",
            "-e", "MODEL_TOKEN=local-development-token",
            "-e", "QFA_DISABLE_STARTERS=1",
            "-e", f"QFA_MAX_STEPS={args.max_steps}",
            "-e", f"QFA_MAX_RESPONSE_TOKENS={args.max_response_tokens}",
            "-e", f"QFA_MODEL_TIMEOUT_SEC={args.model_timeout_sec}",
            "-e", "QFA_RESERVE_SEC=5",
            "-e", "QFA_TRACE_PATH=/eval/trajectory.jsonl",
            "-e", "QFA_RUN_METRICS_PATH=/eval/run_metrics.json",
            "-e", "QFA_SCRATCH_DIR=/eval/scratch",
        ]
        if args.json_mode:
            agent_environment.extend(["-e", "QFA_JSON_MODE=1"])
        agent_command = [
            "docker", "run", "--name", container_name, "--rm",
            *agent_environment,
            "-v", f"{unit.resolve()}:/input:ro",
            "-v", f"{output_dir.resolve()}:/app/output",
            "-v", f"{run_dir.resolve()}:/eval",
            args.agent_image,
            "solve", "--task-dir", "/input", "--out", "/app/output",
        ]
        agent_rc, agent_stdout, agent_stderr, wall_time, timed_out = run_process(
            agent_command,
            timeout=args.task_timeout_sec,
        )
        if timed_out:
            docker_remove(container_name)
        (run_dir / "agent.stdout.log").write_text(agent_stdout, encoding="utf-8")
        (run_dir / "agent.stderr.log").write_text(agent_stderr, encoding="utf-8")

        verifier_rc = 125
        verifier_stdout = ""
        verifier_stderr = ""
        checker = unit / "checks" / "test_outputs.py"
        if checker.exists():
            verifier_command = [
                "docker", "run", "--rm", "--network=none",
                "-e", "OUTPUT_DIR=/app/output",
                "-v", f"{output_dir.resolve()}:/app/output:ro",
                "-v", f"{(unit / 'checks').resolve()}:/checks:ro",
                "-v", f"{(unit / 'environment' / 'data').resolve()}:/app/data:ro",
                "-v", f"{unit.resolve()}:/input:ro",
                args.verifier_image,
                "python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                "/checks/test_outputs.py",
            ]
            verifier_rc, verifier_stdout, verifier_stderr, _, _ = run_process(
                verifier_command,
                timeout=300.0,
            )
        (run_dir / "verifier.stdout.log").write_text(verifier_stdout, encoding="utf-8")
        (run_dir / "verifier.stderr.log").write_text(verifier_stderr, encoding="utf-8")
        verifier_passed, verifier_failed = verifier_counts(verifier_stdout + "\n" + verifier_stderr)
        reward = int(agent_rc == 0 and verifier_rc == 0 and not timed_out)
        metrics = load_json_if_present(run_dir / "run_metrics.json")
        traced = trace_metrics(run_dir / "trajectory.jsonl")
        if timed_out:
            failure_type = "agent_timeout"
        elif metrics.get("status") == "incomplete":
            failure_type = "agent_incomplete"
        elif agent_rc != 0:
            failure_type = "agent_error"
        elif verifier_rc != 0:
            failure_type = "verifier_failure"
        else:
            failure_type = "pass"
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
            "duration_sec": float(metrics.get("duration_sec", wall_time)),
            "steps": int(metrics.get("steps", traced["steps"]) or traced["steps"]),
            "model_calls": int(metrics.get("model_calls", traced["model_calls"]) or traced["model_calls"]),
            "input_tokens": int(metrics.get("input_tokens", traced["input_tokens"]) or traced["input_tokens"]),
            "output_tokens": int(metrics.get("output_tokens", traced["output_tokens"]) or traced["output_tokens"]),
            "starter_used": bool(metrics.get("starter_used", False)),
            "routed_category": metrics.get("routed_category", ""),
            "solver_architecture": metrics.get("solver_architecture", ""),
            "model_name": metrics.get("model_name", args.model_name),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        rows.append(row)
        write_aggregate(experiment_dir, rows)
        print(
            f"  reward={reward} failure={failure_type} calls={row['model_calls']} "
            f"tokens={row['input_tokens']}+{row['output_tokens']} duration={row['duration_sec']:.1f}s",
            flush=True,
        )

    write_aggregate(experiment_dir, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
