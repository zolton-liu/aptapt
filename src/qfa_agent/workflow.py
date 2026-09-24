"""Deterministic workflow controller for the single-model finance agent.

The public finance-agent projects that inspired this module use multiple role
agents or an experiment graph.  Track 1 has a much tighter request budget, so
the roles are represented as explicit states around one House model instead of
as additional model calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .strategies import SolverStrategy
from .types import Action, ToolOutcome


_INSPECTION_TOOLS = {"list_files", "read_file", "search_files"}
_MUTATION_TOOLS = {"write_file", "replace_text", "replace_lines", "copy_file"}


def classify_failure(outcome: ToolOutcome) -> str:
    """Classify tool feedback without spending another model request."""

    text = "\n".join(
        (
            outcome.summary,
            str(outcome.data.get("error", "")),
            str(outcome.data.get("output", "")),
        )
    ).lower()
    if outcome.data.get("timed_out") or "timed out" in text or "timeout" in text:
        return "timeout"
    if "syntaxerror" in text or "syntax validation" in text or "indentationerror" in text:
        return "syntax"
    if "modulenotfounderror" in text or "importerror" in text:
        return "dependency"
    if any(token in text for token in ("nameerror", "unboundlocalerror", "not defined")):
        return "symbol"
    if "filenotfounderror" in text or "path does not exist" in text or "no such file" in text:
        return "path"
    if any(token in text for token in ("keyerror", "usecols", "column", "schema", "header")):
        return "schema"
    if any(token in text for token in ("alignment mismatch", "could not be broadcast", "shapes (")):
        return "alignment"
    if any(token in text for token in ("nan", "non-finite", "infinite", "overflow", "singular", "converge")):
        return "numeric"
    if any(token in text for token in ("typeerror", "valueerror", "dtype", "could not convert")):
        return "type"
    if any(token in text for token in ("assertionerror", "assert ", "self-authored tests failed")):
        return "invariant"
    if any(token in text for token in ("memoryerror", "resource", "killed")):
        return "resource"
    if "protocolerror" in text or "json tool action" in text:
        return "protocol"
    if "output" in text and ("missing" in text or "empty" in text or "failed" in text):
        return "deliverable"
    return "unknown"


def failure_signature(outcome: ToolOutcome) -> str:
    """Return a stable signature so repeated repair loops are visible."""

    raw = "\n".join(
        (
            outcome.summary,
            str(outcome.data.get("error", "")),
            str(outcome.data.get("output", "")),
        )
    )
    # Remove values that normally change between identical attempts.
    normalised = re.sub(r"0x[0-9a-fA-F]+", "0x#", raw)
    normalised = re.sub(r"\b\d+(?:\.\d+)?\b", "#", normalised)
    normalised = normalised[-6000:]
    return hashlib.sha256(normalised.encode("utf-8", errors="replace")).hexdigest()[:16]


@dataclass
class WorkflowState:
    phase: str = "build"
    successful_runs: int = 0
    failed_runs: int = 0
    repair_attempts: int = 0
    validation_passed: bool = False
    last_failure_kind: str = ""
    last_failure_signature: str = ""
    repeated_failure_count: int = 0
    tests_failed: bool = False
    executions_failed: bool = False


class WorkflowController:
    """A low-cost state graph with an explicit final risk gate."""

    def __init__(self, strategy: SolverStrategy):
        self.strategy = strategy
        self.state = WorkflowState()
        self._test_suites: dict[tuple[str, ...], bool] = {}
        self._executions: dict[str, bool] = {}
        self._stage_evidence: dict = {}

    def guard(self, action: Action) -> ToolOutcome | None:
        """Reject only unsafe transitions; guidance handles softer preferences."""

        if action.tool == "finish" and self.state.tests_failed:
            return ToolOutcome(False, "workflow risk gate blocked finish: self-authored tests still failing",
                               {"required_next_step": "Repair the implementation and rerun the original tests. File validation does not prove numerical correctness."})
        if action.tool == "finish" and self.state.executions_failed:
            return ToolOutcome(False, "workflow risk gate blocked finish: execution still failing",
                               {"required_next_step": "Repair and rerun the failed script. Partial files left by a failed run do not prove successful execution."})
        if action.tool == "finish" and not self.state.validation_passed:
            return ToolOutcome(
                False,
                "workflow risk gate blocked finish: validate outputs after the latest mutation",
                {
                    "required_next_step": (
                        "Run the solver or repair it as needed, then call validate_outputs. "
                        "Finish is allowed only after that explicit deterministic gate passes."
                    )
                },
            )
        return None

    def observe(self, action: Action, outcome: ToolOutcome) -> None:
        state = self.state

        if action.tool in _MUTATION_TOOLS:
            if outcome.mutated:
                state.validation_passed = False
                if self._stage_evidence:
                    self._stage_evidence = {**self._stage_evidence, 'stale': True, 'complete': False}
                if self._test_suites:
                    self._test_suites = dict.fromkeys(self._test_suites, False)
                    state.tests_failed = True
            if outcome.ok:
                target = action.arguments.get(
                    "path", action.arguments.get("destination", "")
                )
                state.phase = "execute" if str(target).endswith(".py") else "audit"
            else:
                self._record_failure(outcome)
        elif action.tool == "run_python":
            state.validation_passed = False
            # Missing paths and blocked requests never executed a script.
            # They must not create an impossible execution-recovery gate.
            if 'returncode' in outcome.data:
                self._executions[str(action.arguments.get("script", ""))] = outcome.ok
                state.executions_failed = not all(self._executions.values())
                if isinstance(outcome.data.get('stage_evidence'), dict):
                    from .context import focused_preview
                    self._stage_evidence = focused_preview('stage_evidence', outcome.data['stage_evidence'])
            if outcome.ok:
                state.successful_runs += 1
                state.phase = "audit"
                state.repeated_failure_count = 0
            else:
                state.failed_runs += 1
                self._record_failure(outcome)
        elif action.tool == "run_pytest":
            paths = action.arguments.get("paths", [])
            suite = tuple(sorted(str(path) for path in paths)) if isinstance(paths, list) else (str(paths),)
            self._test_suites[suite] = outcome.ok
            state.tests_failed = not all(self._test_suites.values())
            if outcome.ok:
                state.phase = "audit"
            else:
                self._record_failure(outcome)
        elif action.tool == "validate_outputs":
            state.validation_passed = outcome.ok
            if outcome.ok and (state.tests_failed or state.executions_failed):
                state.phase = "repair"
                state.last_failure_kind = "invariant"
            elif outcome.ok:
                state.phase = "finish"
                state.repeated_failure_count = 0
            else:
                self._record_failure(outcome)
        elif action.tool == "finish" and not outcome.ok:
            state.phase = "audit"
        elif action.tool in _INSPECTION_TOOLS and state.phase == "build":
            state.phase = "build"

        outcome.data["workflow"] = self.snapshot()
        outcome.data["workflow_guidance"] = self.guidance()

    def _record_failure(self, outcome: ToolOutcome) -> None:
        state = self.state
        signature = failure_signature(outcome)
        if signature == state.last_failure_signature:
            state.repeated_failure_count += 1
        else:
            state.repeated_failure_count = 1
        state.last_failure_signature = signature
        state.last_failure_kind = classify_failure(outcome)
        state.repair_attempts += 1
        state.phase = "repair"

    def guidance(self) -> str:
        state = self.state
        if state.phase == "build":
            return "Builder: write the smallest complete executable solver, then run it."
        if state.phase == "execute":
            return "Executor: run the current solver now; do not spend a turn re-reading the task."
        if state.phase == "repair":
            guidance = {
                "syntax": "Repair the exact syntax location from source_context, then rerun.",
                "dependency": "Use the installed finance/Python stack or standard library; remove the unavailable import.",
                "symbol": "Define or pass the missing symbol at the narrowest scope, then rerun.",
                "path": "Resolve the path from TASK_DIR/OUTPUT_DIR and the supplied inventory.",
                "schema": "Align names and types to the bounded input profile and required output contract.",
                "type": "Normalize dtypes/units at the boundary, then rerun the same focused case.",
                "alignment": "Align by the task's date/entity keys before converting to arrays; check identical indexes and never truncate arrays to hide a mismatch.",
                "invariant": "Fix the financial identity that failed; do not weaken the self-test.",
                "numeric": "Add stable edge handling, finite checks, and a bracketed or regularized method.",
                "timeout": "Reduce algorithmic cost and data copies before rerunning.",
                "deliverable": "Create exactly the required non-empty artifacts and schemas.",
            }.get(state.last_failure_kind, "Use the concrete observation to make one bounded repair, then rerun.")
            if state.repeated_failure_count >= 2:
                guidance += (
                    " The same failure recurred: inspect the current scratch source and rewrite the "
                    "smallest faulty function instead of repeating the same patch or run."
                )
            failed_stage = next((s.get('name') for s in self._stage_evidence.get('stages', [])
                                 if s.get('status') == 'failed'), None)
            stage_hint = f' Failed stage: {failed_stage}; repair that function and rerun all stages.' if failed_stage else ''
            return f"Repairer ({state.last_failure_kind}): {guidance}{stage_hint}"
        if state.phase == "audit":
            checks = "; ".join(self.strategy.invariants)
            return (
                "Risk gate: check the produced artifacts against these domain invariants: "
                f"{checks}. Then call validate_outputs."
            )
        return "Portfolio-manager gate: deterministic validation passed; call finish now."

    def snapshot(self) -> dict[str, object]:
        state = self.state
        return {
            "phase": state.phase,
            "successful_runs": state.successful_runs,
            "failed_runs": state.failed_runs,
            "repair_attempts": state.repair_attempts,
            "validation_passed": state.validation_passed,
            "last_failure_kind": state.last_failure_kind,
            "repeated_failure_count": state.repeated_failure_count,
            "tests_failed": state.tests_failed,
            "executions_failed": state.executions_failed,
            "pending_scripts": [path for path, passed in self._executions.items() if not passed],
            "pending_test_suites": [list(paths) for paths, passed in self._test_suites.items() if not passed],
            "solver_stages": self._stage_evidence,
        }

    def as_prompt(self) -> str:
        return json.dumps(self.snapshot(), ensure_ascii=False, sort_keys=True)
