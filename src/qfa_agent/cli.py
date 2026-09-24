"""Official `solve --task-dir ... --out ...` entry point."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

from .agent import AgentConfig, CodingAgent
from .model import ModelError, model_from_environment
from .task import load_task
from .trace import Trajectory
from .workspace import TaskWorkspace


def _write_run_metrics(path_value: str | None, payload: dict[str, object]) -> None:
    if not path_value:
        return
    path = Path(path_value).resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solve", description="Agenthon 2026 T1 minimal agent")
    parser.add_argument("verb", nargs="?", default="solve")
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-budget-sec", type=float,
                        help="Optional outer runtime limit; can only shorten the task card budget")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # Installed console script: `solve --task-dir ...`; Docker entry point:
    # `python -m qfa_agent.cli solve --task-dir ...`.
    if not raw or raw[0].startswith("-"):
        raw.insert(0, "solve")
    args = _parser().parse_args(raw)
    if args.verb != "solve":
        print(f"unsupported verb: {args.verb!r}; expected 'solve'", file=sys.stderr)
        return 2

    metrics_path = os.environ.get("QFA_RUN_METRICS_PATH")
    task_id = "unknown"
    started = time.monotonic()
    try:
        if args.time_budget_sec is not None and (
            not math.isfinite(args.time_budget_sec) or args.time_budget_sec <= 0
        ):
            raise ValueError("time budget must be finite and positive")
        task = load_task(args.task_dir)
        task_id = task.task_id
        output = args.out.resolve(strict=False)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("output directory must be empty to prevent stale deliverables")
        config = AgentConfig.from_environment()
        model = model_from_environment()
        scratch_value = os.environ.get("QFA_SCRATCH_DIR")
        scratch_context = (
            nullcontext(Path(scratch_value).resolve(strict=False))
            if scratch_value
            else tempfile.TemporaryDirectory(prefix="qfa-agent-")
        )
        with scratch_context as temporary:
            scratch = Path(temporary).resolve()
            scratch.mkdir(parents=True, exist_ok=True)
            trace_value = os.environ.get("QFA_TRACE_PATH")
            trace_path = Path(trace_value).resolve(strict=False) if trace_value else scratch / "trajectory.jsonl"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trajectory = Trajectory(trace_path)
            workspace = TaskWorkspace(
                task.root,
                output,
                scratch,
                canaries=task.canaries,
            )
            result = CodingAgent(model, config).solve(
                task, workspace, trajectory,
                deadline=(started + args.time_budget_sec
                          if args.time_budget_sec is not None else started + task.timeout_sec),
            )
        duration_sec = round(time.monotonic() - started, 6)
        _write_run_metrics(
            metrics_path,
            {
                "task_id": task.task_id,
                "status": result.status,
                "message": result.message,
                "context_memory_enabled": config.context_memory_enabled,
                "context_max_tokens": config.context_max_tokens,
                "succeeded": result.succeeded,
                "steps": result.steps,
                "model_calls": result.model_calls,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "starter_used": result.starter_used,
                "routed_category": result.routed_category,
                "solver_architecture": result.solver_architecture,
                "duration_sec": duration_sec,
                "deliverables": list(result.output_validation.files),
                "issues": list(result.output_validation.issues),
                "warnings": list(result.output_validation.warnings),
                "model_name": getattr(model, "name", "unknown"),
            },
        )
        print(
            f"qfa-agent task={task.task_id} status={result.status} "
            f"steps={result.steps} model_calls={result.model_calls} "
            f"deliverables={len(result.output_validation.files)}",
            file=sys.stderr,
        )
        if not result.succeeded:
            for issue in result.output_validation.issues:
                print(f"qfa-agent output issue: {issue}", file=sys.stderr)
            return 1
        return 0
    except (ModelError, OSError, ValueError) as exc:
        _write_run_metrics(
            metrics_path,
            {
                "task_id": task_id,
                "status": "error",
                "succeeded": False,
                "duration_sec": round(time.monotonic() - started, 6),
                "error_type": type(exc).__name__,
                "model_name": os.environ.get("MODEL_NAME", "unknown"),
            },
        )
        # Never print task contents, proxy settings, or model responses here.
        print(f"qfa-agent failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
