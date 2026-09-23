"""Bounded, content-minimising trajectory log."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .types import Action, ToolOutcome


class Trajectory:
    def __init__(self, path: Path, *, max_bytes: int = 2_000_000):
        self.path = path
        self.max_bytes = max_bytes
        self.started = time.monotonic()
        self._written = 0

    def _append(self, event: dict[str, Any]) -> None:
        event = {"elapsed_sec": round(time.monotonic() - self.started, 3), **event}
        line = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode()
        if self._written + len(line) > self.max_bytes:
            return
        with self.path.open("ab") as handle:
            handle.write(line)
        self._written += len(line)

    def model_call(self, step: int, content: str | dict[str, Any], input_tokens: int, output_tokens: int) -> None:
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True).encode()
        self._append(
            {
                "event": "model_call",
                "step": step,
                "content_bytes": len(encoded),
                "content_sha256": hashlib.sha256(encoded).hexdigest(),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        )

    def context(self, step: int, stats: dict[str, Any]) -> None:
        self._append({"event": "context", "step": step, **stats})

    def route(
        self,
        category: str,
        architecture: str,
        source: str,
        confidence: float,
    ) -> None:
        self._append(
            {
                "event": "route",
                "category": category,
                "architecture": architecture,
                "source": source,
                "confidence": confidence,
            }
        )

    def action(self, step: int, action: Action, outcome: ToolOutcome) -> None:
        encoded_args = json.dumps(action.arguments, ensure_ascii=False, sort_keys=True).encode()
        workflow = outcome.data.get("workflow", {})
        self._append(
            {
                "event": "tool",
                "step": step,
                "tool": action.tool,
                "argument_keys": sorted(action.arguments),
                "arguments_sha256": hashlib.sha256(encoded_args).hexdigest(),
                "ok": outcome.ok,
                "mutated": outcome.mutated,
                "summary": outcome.summary[:500],
                "workflow_phase": workflow.get("phase") if isinstance(workflow, dict) else None,
                "failure_kind": workflow.get("last_failure_kind") if isinstance(workflow, dict) else None,
            }
        )

    def finish(self, status: str, files: tuple[str, ...]) -> None:
        self._append({"event": "finish", "status": status, "files": list(files)})
