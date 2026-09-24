"""Bounded subprocess execution without a shell."""

from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

from .types import ProcessResult


_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "CREDENTIAL",
    "AWS_",
    "AZURE_",
    "GOOGLE_APPLICATION_CREDENTIALS",
)


def _child_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(marker in upper for marker in _SENSITIVE_ENV_MARKERS):
            continue
        # Generated programs never need model/proxy configuration.
        if upper in {
            "MODEL_ENDPOINT",
            "MODEL_NAME",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "ALL_PROXY",
            "NO_PROXY",
            "QFA_EXPERIENCE_DIR",
        }:
            continue
        env[key] = value
    env.update(
        {
            "PYTHONHASHSEED": os.environ.get("QFBENCH_SEED", "0"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
            "LANG": "C.UTF-8",
            "MPLBACKEND": "Agg",
        }
    )
    if extra:
        env.update(extra)
    return env


def _read_head_tail(path: Path, limit: int) -> tuple[str, int]:
    size = path.stat().st_size
    if size <= limit:
        data = path.read_bytes()
        return data.decode("utf-8", errors="replace"), 0
    head_size = max(1, limit // 3)
    tail_size = max(1, limit - head_size)
    with path.open("rb") as handle:
        head = handle.read(head_size)
        handle.seek(-tail_size, os.SEEK_END)
        tail = handle.read(tail_size)
    omitted = max(0, size - len(head) - len(tail))
    marker = f"\n... <{omitted} output bytes omitted> ...\n".encode()
    data = head + marker + tail
    return data.decode("utf-8", errors="replace"), omitted


def run_bounded(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_sec: float,
    output_limit_bytes: int = 16_000,
    extra_env: Mapping[str, str] | None = None,
) -> ProcessResult:
    if not argv or not all(isinstance(arg, str) and arg for arg in argv):
        raise ValueError("argv must be a non-empty sequence of non-empty strings")
    started = time.monotonic()
    timed_out = False

    with tempfile.NamedTemporaryFile(prefix="qfa-process-", suffix=".log", delete=False) as stream:
        log_path = Path(stream.name)
        try:
            process = subprocess.Popen(
                list(argv),
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
                env=_child_env(extra_env),
                shell=False,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=max(0.1, timeout_sec))
            except subprocess.TimeoutExpired:
                timed_out = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    returncode = process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    returncode = process.wait()
        except OSError as exc:
            returncode = 127
            stream.write(f"unable to start process: {exc}\n".encode())
            stream.flush()
        finally:
            stream.flush()

    try:
        output, omitted = _read_head_tail(log_path, max(512, output_limit_bytes))
    finally:
        log_path.unlink(missing_ok=True)
    return ProcessResult(
        argv=tuple(argv),
        returncode=returncode,
        output=output,
        duration_sec=round(time.monotonic() - started, 3),
        timed_out=timed_out,
        truncated_bytes=omitted,
    )
