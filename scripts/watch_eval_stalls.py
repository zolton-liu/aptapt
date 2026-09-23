#!/usr/bin/env python3
"""Terminate only a stalled per-task agent child of a long public evaluation.

The local OpenAI-compatible streaming client can occasionally remain blocked
after its configured request timeout.  This watchdog leaves the evaluator
alone and sends SIGTERM only to its current ``qfa_agent.cli solve`` child when
that task's trajectory has not changed for longer than the supplied limit.
The evaluator then records the task as an agent error and advances normally.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def evaluator_child(parent_pid: int) -> tuple[int, str] | None:
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    for raw_line in result.stdout.splitlines():
        fields = raw_line.strip().split(maxsplit=2)
        if len(fields) != 3:
            continue
        pid_text, ppid_text, command = fields
        if int(ppid_text) != parent_pid:
            continue
        if "qfa_agent.cli solve" in command:
            return int(pid_text), command
    return None


def trajectory_from_command(command: str, experiment_dir: Path) -> Path | None:
    marker = " --out "
    if marker not in command:
        return None
    output_text = command.split(marker, 1)[1].split(" --", 1)[0].strip()
    output_dir = Path(output_text)
    try:
        output_dir.resolve().relative_to(experiment_dir.resolve())
    except ValueError:
        return None
    return output_dir.parent / "trajectory.jsonl"


def inspect_once(
    parent_pid: int,
    experiment_dir: Path,
    stale_sec: float,
    observations: dict[Path, tuple[tuple[int, int], float]],
) -> bool:
    child = evaluator_child(parent_pid)
    if child is None:
        print("watchdog: no active solve child", flush=True)
        return False
    child_pid, command = child
    trajectory = trajectory_from_command(command, experiment_dir)
    if trajectory is None or not trajectory.exists():
        print(f"watchdog: child={child_pid} has no eligible trajectory", flush=True)
        return False
    stat = trajectory.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    now = time.monotonic()
    previous_signature, last_progress = observations.get(trajectory, (signature, now))
    if signature != previous_signature:
        last_progress = now
    observations.clear()
    observations[trajectory] = (signature, last_progress)
    stalled_for = now - last_progress
    task_id = trajectory.parent.name
    print(
        f"watchdog: task={task_id} child={child_pid} "
        f"observed_stall={stalled_for:.1f}s",
        flush=True,
    )
    if stalled_for <= stale_sec:
        return False
    os.kill(child_pid, signal.SIGTERM)
    print(
        f"watchdog: terminated stalled task={task_id} child={child_pid} "
        f"after {stalled_for:.1f}s of continuously observed monotonic time "
        "without trajectory progress",
        flush=True,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--stale-sec", type=float, default=420.0)
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.is_dir():
        raise SystemExit(f"missing experiment directory: {experiment_dir}")
    if args.stale_sec <= 0 or args.poll_sec <= 0:
        raise SystemExit("stale and poll intervals must be positive")

    if args.once:
        inspect_once(args.parent_pid, experiment_dir, args.stale_sec, {})
        return 0

    observations: dict[Path, tuple[tuple[int, int], float]] = {}
    while process_exists(args.parent_pid):
        inspect_once(args.parent_pid, experiment_dir, args.stale_sec, observations)
        time.sleep(args.poll_sec)
    print(f"watchdog: evaluator parent {args.parent_pid} exited", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
