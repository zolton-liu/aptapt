"""Small data contracts shared by the agent core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Message = dict[str, str]


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutcome:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    mutated: bool = False

    def as_observation(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "summary": self.summary,
            "data": self.data,
            "mutated": self.mutated,
        }


@dataclass(frozen=True)
class ProcessResult:
    argv: tuple[str, ...]
    returncode: int
    output: str
    duration_sec: float
    timed_out: bool = False
    truncated_bytes: int = 0

    @property
    def passed(self) -> bool:
        return self.returncode == 0 and not self.timed_out


@dataclass(frozen=True)
class OutputValidation:
    ok: bool
    files: tuple[str, ...]
    issues: tuple[str, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    status: str
    steps: int
    model_calls: int
    output_validation: OutputValidation
    message: str
    input_tokens: int = 0
    output_tokens: int = 0
    starter_used: bool = False
    routed_category: str = ""
    solver_architecture: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "completed" and self.output_validation.ok
