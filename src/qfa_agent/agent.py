"""Single-agent, single-action, bounded T1 solve loop."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass

from .context import ContextBudgetError, ContextManager, estimate_tokens
from .memory import WorkingMemory
from .model import Model, ModelError
from .prompts import system_prompt, user_prompt
from .protocol import ProtocolError, compact_action, parse_action
from .repair import repair_context
from .starters import starter_for, trusted_operator_program_for
from .strategies import route_task
from .task import TaskSpec, expected_output_files
from .trace import Trajectory
from .types import Action, AgentResult, Message, ToolOutcome
from .workflow import WorkflowController
from .workspace import TaskWorkspace, ToolRouter, WorkspaceError


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 20
    max_model_calls: int = 25
    max_format_errors: int = 3
    max_model_errors: int = 2
    max_identical_actions: int = 2
    max_python_runs: int = 8
    max_pytest_runs: int = 3
    model_call_timeout_sec: float = 300.0
    input_token_budget: int = 1_000_000
    output_token_budget: int = 100_000
    reserve_sec: float = 45.0
    starters_enabled: bool = True
    context_memory_enabled: bool = True
    context_max_tokens: int = 16000
    observation_max_chars: int = 6000
    auto_complete_valid_run: bool = False

    @classmethod
    def from_environment(cls) -> "AgentConfig":
        disable_starters = os.environ.get("QFA_DISABLE_STARTERS", "").strip().lower()
        return cls(
            max_steps=min(25, int(os.environ.get("QFA_MAX_STEPS", "20"))),
            max_model_calls=min(25, int(os.environ.get("QFA_MAX_MODEL_CALLS", "25"))),
            max_format_errors=int(os.environ.get("QFA_MAX_FORMAT_ERRORS", "3")),
            max_model_errors=int(os.environ.get("QFA_MAX_MODEL_ERRORS", "2")),
            max_identical_actions=int(os.environ.get("QFA_MAX_IDENTICAL_ACTIONS", "2")),
            max_python_runs=int(os.environ.get("QFA_MAX_PYTHON_RUNS", "8")),
            max_pytest_runs=int(os.environ.get("QFA_MAX_PYTEST_RUNS", "3")),
            model_call_timeout_sec=float(os.environ.get("QFA_MODEL_TIMEOUT_SEC", "300")),
            input_token_budget=min(
                1_000_000, int(os.environ.get("QFA_INPUT_TOKEN_BUDGET", "1000000"))
            ),
            output_token_budget=min(
                100_000, int(os.environ.get("QFA_OUTPUT_TOKEN_BUDGET", "100000"))
            ),
            reserve_sec=float(os.environ.get("QFA_RESERVE_SEC", "45")),
            starters_enabled=disable_starters not in {"1", "true", "yes", "on"},
            context_memory_enabled=os.environ.get("QFA_CONTEXT_MEMORY", "1").lower()
            not in {"0", "false", "off"},
            context_max_tokens=max(1024, int(os.environ.get("QFA_CONTEXT_MAX_TOKENS", "16000"))),
            observation_max_chars=max(1500, int(os.environ.get("QFA_OBSERVATION_MAX_CHARS", "6000"))),
            auto_complete_valid_run=os.environ.get("QFA_AUTO_COMPLETE_VALID_RUN", "0").lower()
            in {"1", "true", "yes", "on"},
        )


class CodingAgent:
    def __init__(self, model: Model, config: AgentConfig):
        self.model = model
        self.config = config

    @staticmethod
    def _required_high_level_operator(task: TaskSpec) -> str | None:
        """Return a policy-mandated operator for fragile, fully-covered families.

        The model still owns input-path selection and the final execution plan;
        this gate prevents it from silently discarding a tested numerical
        operator and retyping a long, failure-prone implementation.
        """

        lowered = task.safe_instruction.lower()
        if "cir short-rate" in lowered or "cox-ingersoll-ross" in lowered:
            return "write_cir_calibration_outputs"
        if "put-call parity" in lowered and "synthetic_forward_bid" in lowered:
            return "write_put_call_parity_audit"
        if "ohlc realized volatility estimators" in lowered and "yang-zhang" in lowered:
            return "write_ohlc_volatility_outputs"
        if "arithmetic asian options" in lowered and "curran" in lowered and "levy" in lowered:
            return "write_asian_option_outputs"
        if "cliquet" in lowered and "forward-start" in lowered:
            return "write_cliquet_outputs"
        return None

    @staticmethod
    def _observation(outcome: ToolOutcome) -> str:
        return json.dumps(
            {"type": "tool_observation", **outcome.as_observation()},
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def _postprocess_python_outcome(
        workspace: TaskWorkspace,
        router: ToolRouter,
        action: Action,
        outcome: ToolOutcome,
        required_files: tuple[str, ...],
    ) -> ToolOutcome:
        """Attach focused repair context and deterministic artifact feedback."""

        script_path = str(action.arguments.get("script", ""))
        if not outcome.ok:
            raw_output = str(outcome.data.get("output", ""))
            try:
                auto_repair = workspace.repair_known_runtime_issue(script_path, raw_output)
            except (WorkspaceError, OSError, UnicodeError, ValueError):
                auto_repair = None
            if auto_repair is not None:
                retried = router.dispatch(action)
                retried.data["auto_repair"] = auto_repair
                retried.data["auto_repair_note"] = (
                    "A narrow runtime-authorized repair was applied and the script was "
                    "rerun without another model request."
                )
                outcome = retried
                raw_output = str(outcome.data.get("output", ""))
            if os.environ.get('QFA_REPAIR_CONTEXT', '1').lower() not in {'0', 'false', 'off'}:
                try:
                    packet = repair_context(workspace, raw_output)
                    outcome.data['repair_context'] = packet
                    if packet.get('location_verified'):
                        outcome.data['source_context'] = packet['source_context']
                except (OSError, ValueError, WorkspaceError):
                    pass
            if not outcome.ok:
                outcome.data["required_next_step"] = (
                    "Use the participant-owned repair_context path/line and related variable definitions/call sites; "
                    "library line numbers do not belong to solve.py. Make one precise repair, then rely on "
                    "the controller auto-run. Do not browse unrelated inputs or repeat the failed run."
                )

        if outcome.ok:
            outcome.data["artifact_manifest"] = workspace.output_manifest()
            validation = workspace.validate_outputs(required_files)
            outcome.data["output_contract"] = {
                "ok": validation.ok,
                "files": list(validation.files),
                "issues": list(validation.issues),
                "warnings": list(validation.warnings),
            }
        return outcome

    @staticmethod
    def _candidate_solver_digest(
        workspace: TaskWorkspace, action: Action
    ) -> str | None:
        """Preview a solve.py edit so an A→B→A repair cycle can be blocked."""

        if str(action.arguments.get("path", "")).lower() != "scratch/solve.py":
            return None
        try:
            if action.tool == "write_file":
                content = action.arguments.get("content")
                if not isinstance(content, str):
                    return None
                candidate = content
            elif action.tool == "replace_text":
                old = action.arguments.get("old")
                new = action.arguments.get("new")
                if not isinstance(old, str) or not old or not isinstance(new, str):
                    return None
                _, path = workspace.resolve("scratch/solve.py")
                current = path.read_text(encoding="utf-8")
                if current.count(old) != 1:
                    return None
                candidate = current.replace(old, new, 1)
            elif action.tool == 'replace_function':
                from .function_edit import replace_function_source
                _, path = workspace.resolve('scratch/solve.py')
                candidate = replace_function_source(path.read_bytes().decode('utf-8'),
                    action.arguments.get('name'), action.arguments.get('content'),
                    action.arguments.get('expected_sha256'))
            else:
                return None
        except (WorkspaceError, OSError, UnicodeError, ValueError, SyntaxError):
            return None
        return hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    @staticmethod
    def _current_solver_digest(workspace: TaskWorkspace) -> str | None:
        try:
            _, path = workspace.resolve("scratch/solve.py")
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except (WorkspaceError, OSError):
            return None

    @staticmethod
    def _model_context(
        messages: list[Message],
        *,
        workflow: WorkflowController,
        workspace: TaskWorkspace,
        action_history: list[dict[str, object]],
        failure_memory: list[dict[str, object]],
    ) -> list[Message]:
        """Bound repair context while retaining deterministic task state.

        The system prompt already contains the complete instruction, inventory,
        and data profile. After four tool turns, older full observations mostly
        repeat source and tracebacks. Replace them with a compact checkpoint and
        keep the three most recent action/observation pairs, which preserves the
        exact source slice needed for a focused repair.
        """

        if len(messages) <= 10:
            return messages
        checkpoint = {
            "type": "deterministic_progress_checkpoint",
            "note": (
                "Older detailed tool turns were compacted. Do not repeat an old action. "
                "Use read_file if exact current source is needed."
            ),
            "workflow": workflow.snapshot(),
            "recent_action_results": action_history[-10:],
            "known_failure_constraints": failure_memory[-8:],
            "scratch_files": list(workspace.inventory("scratch")[:80]),
            "output_manifest": workspace.output_manifest()[:40],
        }
        return [
            messages[0],
            messages[1],
            {"role": "user", "content": json.dumps(checkpoint, ensure_ascii=False, sort_keys=True)},
            *messages[-6:],
        ]

    def solve(self, task: TaskSpec, workspace: TaskWorkspace, trajectory: Trajectory,
              *, deadline: float | None = None) -> AgentResult:
        started = time.monotonic()
        deadline = min(deadline, started + task.timeout_sec) if deadline is not None else started + task.timeout_sec
        inventory = workspace.inventory("input")
        required_files = expected_output_files(task.safe_instruction)
        input_profile = workspace.input_profile(inventory)
        routing = route_task(task)
        strategy = routing.strategy
        trajectory.route(
            strategy.category,
            strategy.architecture,
            routing.source,
            routing.confidence,
        )
        strict_verification = os.environ.get('QFA_VERIFICATION_REQUIRED', '0') == '1'
        if strict_verification and os.environ.get('QFA_STAGE_PIPELINE', '1').lower() in {'0', 'false', 'off'}:
            raise ValueError('QFA_VERIFICATION_REQUIRED=1 requires QFA_STAGE_PIPELINE enabled')
        starter = starter_for(task, inventory) if self.config.starters_enabled and not strict_verification else None
        starter_note = ""
        if starter is not None:
            workspace.write_file(starter.path, starter.content)
            starter_outcome = workspace.run_python(starter.path, [], min(
                120.0, max(0.1, deadline - time.monotonic() - self.config.reserve_sec)))
            trajectory.action(
                0,
                Action("run_python", {"script": starter.path, "source": "category_starter"}),
                starter_outcome,
            )
            starter_note = (
                f"A deterministic category starter was generated at {starter.path} for "
                f"{starter.purpose} and executed once ({starter_outcome.summary}). "
                "Inspect its code and current output before changing either. Preserve a valid "
                "starter result; use the model to audit, test, or repair it rather than replacing "
                "it without evidence."
            )
            # Category starters are complete, deterministic solvers rather than examples.
            # Avoid spending the task budget on an unnecessary model review once one has
            # executed cleanly and produced structurally valid deliverables.  A failed or
            # incomplete starter still falls through to the normal repair loop below.
            if starter_outcome.ok:
                validation = workspace.validate_outputs(required_files)
                if validation.ok:
                    trajectory.finish("completed", validation.files)
                    return AgentResult(
                        status="completed",
                        steps=0,
                        model_calls=0,
                        output_validation=validation,
                        message="deterministic category starter completed and validation passed",
                        starter_used=True,
                        routed_category=strategy.category,
                        solver_architecture=strategy.architecture,
                    )
        # A full deterministic adapter is a starter regardless of its name.
        # Respect the ablation switch instead of reporting a zero-call solve as
        # starter-disabled model performance.
        operator_program = (
            trusted_operator_program_for(task, inventory)
            if self.config.starters_enabled and not strict_verification else None
        )
        operator_note = ""
        if operator_program is not None:
            workspace.write_file(
                operator_program.path, operator_program.content, overwrite=True
            )
            operator_outcome = workspace.run_python(
                operator_program.path, [], min(180.0, max(
                    0.1, deadline - time.monotonic() - self.config.reserve_sec))
            )
            trajectory.action(
                0,
                Action(
                    "run_python",
                    {
                        "script": operator_program.path,
                        "source": "trusted-finance-operator",
                    },
                ),
                operator_outcome,
            )
            if operator_outcome.ok:
                validation = workspace.validate_outputs(required_files)
                if validation.ok:
                    trajectory.finish("completed", validation.files)
                    return AgentResult(
                        status="completed",
                        steps=0,
                        model_calls=0,
                        output_validation=validation,
                        message=(
                            "trusted finance operator completed and validation passed"
                        ),
                        starter_used=True,
                        routed_category=strategy.category,
                        solver_architecture=strategy.architecture,
                    )
            operator_note = (
                f"A trusted finance operator adapter was attempted at "
                f"{operator_program.path} for {operator_program.purpose}, but did not "
                f"complete the output contract ({operator_outcome.summary}). Inspect and "
                "repair that adapter instead of reimplementing its numerical kernel."
            )
        messages: list[Message] = [
            {
                "role": "system",
                "content": system_prompt(
                    task,
                    inventory,
                    input_profile,
                    routing,
                    " ".join(note for note in (starter_note, operator_note) if note),
                ),
            },
            {"role": "user", "content": user_prompt(task)},
        ]
        router = ToolRouter(
            workspace,
            max_python_runs=min(self.config.max_python_runs, strategy.max_python_runs),
            max_pytest_runs=min(self.config.max_pytest_runs, strategy.max_pytest_runs),
            expected_files=required_files,
            deadline=deadline - self.config.reserve_sec,
        )
        workflow = WorkflowController(strategy)
        context = ContextManager(workspace, max_tokens=self.config.context_max_tokens,
                                 observation_chars=self.config.observation_max_chars)
        memory = WorkingMemory(workspace, task.instruction_sha256, strategy.category)
        stop_reason = ""
        format_errors = 0
        consecutive_model_errors = 0
        model_calls = 0
        input_tokens = 0
        output_tokens = 0
        last_fingerprint = ""
        identical_count = 0
        failed_edit_fingerprints: dict[str, int] = {}
        seen_solver_digests: set[str] = set()
        last_executed_solver_digest: str | None = None
        last_solver_execution_failed = False
        action_history: list[dict[str, object]] = []
        failure_memory: list[dict[str, object]] = []
        inspection_actions = 0
        inspection_cache: dict[str, ToolOutcome] = {}
        actions_without_mutation: dict[str, int] = {}
        latest_execution_evidence: dict[str, object] = {}
        executable_drafted = False
        solver_dirty = False
        required_operator = self._required_high_level_operator(task)
        effective_steps = min(
            self.config.max_steps,
            self.config.max_model_calls,
            strategy.max_model_calls,
            25,
        )

        for step in range(1, effective_steps + 1):
            remaining = deadline - time.monotonic() - self.config.reserve_sec
            if remaining <= 0:
                break
            if input_tokens >= self.config.input_token_budget or output_tokens >= self.config.output_token_budget:
                break
            if self.config.context_memory_enabled:
                checkpoint = memory.checkpoint(
                    workflow={**workflow.snapshot(), "guidance": workflow.guidance()},
                    required_files=required_files,
                    solver_digest=self._current_solver_digest(workspace),
                    budget={"remaining_seconds": round(remaining, 1),
                            "remaining_model_calls": effective_steps - model_calls,
                            "remaining_python_runs": router.max_python_runs - router.python_runs,
                            "remaining_input_tokens": self.config.input_token_budget - input_tokens},
                )
                memory.save(checkpoint)
                try:
                    model_messages, context_stats = context.build(messages, checkpoint)
                except ContextBudgetError as exc:
                    stop_reason = str(exc)
                    trajectory.context(step, {"error": stop_reason})
                    break
                trajectory.context(step, context_stats)
            else:
                model_messages = self._model_context(
                    messages, workflow=workflow, workspace=workspace,
                    action_history=action_history, failure_memory=failure_memory,
                )
            prompt_estimate = estimate_tokens(model_messages)
            if input_tokens + prompt_estimate > self.config.input_token_budget:
                break
            model_calls += 1
            try:
                reply = self.model.complete(
                    model_messages,
                    timeout_sec=min(self.config.model_call_timeout_sec, max(0.1, remaining)),
                )
            except ModelError as exc:
                consecutive_model_errors += 1
                retry_outcome = ToolOutcome(
                    False,
                    f"transient model call failed: {type(exc).__name__}",
                    {
                        "attempt": consecutive_model_errors,
                        "max_attempts": self.config.max_model_errors,
                        "required_next_step": "Retry the same deterministic context without changing workspace state.",
                    },
                )
                trajectory.action(step, Action("model_retry", {}), retry_outcome)
                if self.config.context_memory_enabled:
                    memory.observe(step, Action("model_retry", {}), retry_outcome,
                                   self._current_solver_digest(workspace))
                # Endpoint timeouts may have consumed input even without usage
                # metadata. Reserve the estimate instead of treating them as free.
                input_tokens += prompt_estimate
                action_history.append(
                    {
                        "step": step,
                        "tool": "model_retry",
                        "ok": False,
                        "summary": retry_outcome.summary,
                        "failure_kind": "model_service",
                    }
                )
                if consecutive_model_errors >= self.config.max_model_errors:
                    break
                continue
            if consecutive_model_errors and self.config.context_memory_enabled:
                memory.observe(step, Action("model_retry", {}),
                               ToolOutcome(True, "model endpoint recovered"),
                               self._current_solver_digest(workspace))
            consecutive_model_errors = 0
            encoded_reply = json.dumps(reply.content, ensure_ascii=False).encode("utf-8")
            input_tokens += max(0, reply.input_tokens) or prompt_estimate
            output_tokens += max(0, reply.output_tokens) or math.ceil(len(encoded_reply) / 3)
            trajectory.model_call(step, reply.content, reply.input_tokens, reply.output_tokens)
            if reply.finish_reason.lower() == "length":
                format_errors += 1
                messages.append({"role": "assistant", "content": "<response truncated at model token limit>"})
                messages.append(
                    {
                        "role": "user",
                        "content": self._observation(
                            ToolOutcome(
                                False,
                                "model response was truncated before a complete JSON action",
                                {
                                    "required_next_step": (
                                        "Return one compact action under 1200 tokens. Use replace_lines for a "
                                        "small repair; do not rewrite the entire solver in this turn."
                                    )
                                },
                            )
                        ),
                    }
                )
                if format_errors >= self.config.max_format_errors:
                    break
                continue
            try:
                action = parse_action(reply.content)
            except ProtocolError as exc:
                format_errors += 1
                messages.append({"role": "assistant", "content": "<invalid structured action>"})
                messages.append(
                    {
                        "role": "user",
                        "content": self._observation(
                            ToolOutcome(False, f"ProtocolError: {exc}. Return exactly one JSON tool action.")
                        ),
                    }
                )
                if format_errors >= self.config.max_format_errors:
                    break
                continue

            encoded = json.dumps(
                {"tool": action.tool, "arguments": action.arguments},
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
            fingerprint = hashlib.sha256(encoded).hexdigest()
            actions_without_mutation[fingerprint] = actions_without_mutation.get(fingerprint, 0) + 1
            if fingerprint == last_fingerprint:
                identical_count += 1
            else:
                identical_count = 1
                last_fingerprint = fingerprint
            guarded = workflow.guard(action)
            candidate_solver_digest = self._candidate_solver_digest(workspace, action)
            current_solver_digest = self._current_solver_digest(workspace)
            # A failed edit is invalid for the observed target version, not
            # forever: a later genuine edit may make its match/range valid.
            edit_version = None
            if action.tool in {'write_file', 'replace_text', 'replace_lines', 'copy_file', 'replace_function'}:
                try:
                    _, edit_target = workspace.resolve(
                        str(action.arguments.get('path', action.arguments.get('destination', ''))),
                        write=True, must_exist=False)
                    edit_version = hashlib.sha256(edit_target.read_bytes()).hexdigest() if edit_target.is_file() else 'missing'
                except (WorkspaceError, OSError, ValueError):
                    edit_version = 'unresolved'
            failed_edit_key = f'{fingerprint}:{edit_version}'
            is_inspection = action.tool in {"list_files", "read_file", "search_files"}
            if action.tool == "copy_file" and not str(
                action.arguments.get("destination", "")
            ).lower().endswith(".py"):
                is_inspection = True
            if action.tool == "write_file":
                staged_path = str(action.arguments.get("path", "")).lower()
                if staged_path.startswith("scratch/") and not staged_path.endswith(".py"):
                    is_inspection = True
            if (
                action.tool == "write_file"
                and str(action.arguments.get("path", "")).lower().endswith("solve.py")
                and required_operator is not None
                and required_operator not in str(action.arguments.get("content", ""))
            ):
                outcome = ToolOutcome(
                    False,
                    f"this task family requires the tested {required_operator} operator",
                    {
                        "required_operator": required_operator,
                        "required_next_step": (
                            "Rewrite scratch/solve.py as a short adapter that imports and calls "
                            f"qfa_agent.finance_ops.{required_operator}; do not reimplement its formulas."
                        ),
                    },
                )
            elif guarded is not None:
                outcome = guarded
            elif (
                action.tool == "run_python"
                and str(action.arguments.get("script", "")).lower() == "scratch/solve.py"
                and current_solver_digest is None
            ):
                outcome = ToolOutcome(
                    False,
                    "cannot run before scratch/solve.py exists",
                    {
                        "required_next_step": (
                            "Write a complete scratch/solve.py now from the supplied instruction, "
                            "inventory and data profile."
                        )
                    },
                )
            elif is_inspection and fingerprint in inspection_cache:
                # Context compaction can evict a previous read. Return its
                # actual evidence instead of claiming it is still visible.
                outcome = copy.deepcopy(inspection_cache[fingerprint])
                outcome.data.update(cached=True, required_next_step=(
                    "This is the unchanged cached evidence. Make a concrete repair now; "
                    "another identical request without a mutation will not add information."))
            elif (
                candidate_solver_digest is not None
                and candidate_solver_digest in seen_solver_digests
            ):
                outcome = ToolOutcome(
                    False,
                    "solver edit blocked because it restores a previously executed source version",
                    {
                        "cycle_detected": True,
                        "required_next_step": (
                            "The repair is oscillating between old versions. Re-read the current traceback "
                            "and rewrite scratch/solve.py once with all known compatibility fixes combined."
                        ),
                    },
                )
            elif (
                action.tool == "run_python"
                and str(action.arguments.get("script", "")).lower() == "scratch/solve.py"
                and last_solver_execution_failed
                and current_solver_digest is not None
                and current_solver_digest == last_executed_solver_digest
            ):
                outcome = ToolOutcome(
                    False,
                    "unchanged solve.py already failed; rerunning it cannot make progress",
                    {
                        **latest_execution_evidence,
                        "required_next_step": (
                            "Use the latest traceback/source_context to edit the solver before another run."
                        )
                    },
                )
            elif action.tool in {"write_file", "replace_text", "replace_lines", "copy_file", "replace_function"} and failed_edit_fingerprints.get(failed_edit_key, 0):
                outcome = ToolOutcome(
                    False,
                    "this exact edit already failed earlier; repeating it cannot change the file",
                    {
                        "prior_failures": failed_edit_fingerprints[failed_edit_key],
                        "required_next_step": (
                            "Use candidate_contexts/source_context to make the match unique, or rewrite "
                            "the smallest complete function/file with real source content."
                        ),
                    },
                )
            elif identical_count > self.config.max_identical_actions:
                outcome = ToolOutcome(
                    False,
                    "identical action repeated without progress; inspect the observation and choose another action",
                )
            elif action.tool == "finish":
                validation = workspace.validate_outputs(required_files)
                if validation.ok:
                    trajectory.finish("completed", validation.files)
                    return AgentResult(
                        status="completed",
                        steps=step,
                        model_calls=model_calls,
                        output_validation=validation,
                        message="model finished and generic output checks passed",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        starter_used=starter is not None or operator_program is not None,
                        routed_category=strategy.category,
                        solver_architecture=strategy.architecture,
                    )
                outcome = ToolOutcome(
                    False,
                    "cannot finish because generic output checks failed",
                    {"issues": list(validation.issues), "files": list(validation.files)},
                )
            else:
                if is_inspection and not executable_drafted and (
                    inspection_actions >= strategy.max_inspection_actions
                    or step >= strategy.draft_by_step
                ):
                    outcome = ToolOutcome(
                        False,
                        (
                            f"{strategy.category} inspection budget exhausted; write a complete "
                            "executable draft now using the files and conventions already inspected"
                        ),
                        {
                            "category": strategy.category,
                            "architecture": strategy.architecture,
                            "inspection_actions": inspection_actions,
                        },
                    )
                else:
                    outcome = router.dispatch(action)
                    if is_inspection:
                        inspection_actions += 1
                        if outcome.ok and not outcome.mutated:
                            inspection_cache[fingerprint] = copy.deepcopy(outcome)
                    if outcome.mutated:
                        # Files can also change through execution, not only
                        # direct source edits. Never replay stale output reads.
                        inspection_cache.clear()
                        # A hypothesis/observation file does not repair code or
                        # change the data. Do not let note-taking reset stalls.
                        if action.tool != 'revise_method':
                            actions_without_mutation.clear()
                    if action.tool in {"write_file", "replace_text", "replace_lines", "copy_file", "replace_function"}:
                        path = str(
                            action.arguments.get(
                                "path", action.arguments.get("destination", "")
                            )
                        ).lower()
                        if path.endswith(".py"):
                            executable_drafted = True
                            if outcome.mutated:
                                solver_dirty = True
                                # A source change invalidates previously cached
                                # reads of that source.  Re-reading the same line
                                # range is then progress, not a duplicate loop.
                                inspection_cache.clear()
                    if action.tool == "run_python":
                        outcome = self._postprocess_python_outcome(
                            workspace, router, action, outcome, required_files
                        )
                    if action.tool in {"replace_text", "replace_lines", "replace_function"} and not outcome.ok:
                        outcome.data["required_next_step"] = (
                            "Read the current source and hash. Prefer replace_function for one complete "
                            "top-level function, or use exact replace_text/replace_lines for syntax repairs."
                        )
                    if action.tool == "run_python":
                        solver_dirty = False

            if action.tool in {"run_pytest", "validate_outputs"} and outcome.ok:
                outcome.data["artifact_manifest"] = workspace.output_manifest()
            workflow.observe(action, outcome)
            if (
                action.tool == "run_python"
                and str(action.arguments.get("script", "")).lower() == "scratch/solve.py"
                and "returncode" in outcome.data
            ):
                last_executed_solver_digest = self._current_solver_digest(workspace)
                last_solver_execution_failed = not outcome.ok
                if last_executed_solver_digest is not None:
                    seen_solver_digests.add(last_executed_solver_digest)
            controller_events: list[tuple[Action, ToolOutcome]] = []

            # A complete solve.py write was almost always followed by a model
            # turn whose only action was run_python. Execute it immediately to
            # preserve scarce model calls for semantic repairs.
            if (
                action.tool in {"write_file", "replace_text", "replace_lines", "replace_function"}
                and outcome.ok
                and str(action.arguments.get("path", "")).lower() == "scratch/solve.py"
            ):
                auto_action = Action(
                    "run_python",
                    {
                        "script": "scratch/solve.py",
                        "args": [],
                        "timeout_sec": min(180.0, max(0.1, deadline - time.monotonic() - self.config.reserve_sec)),
                        "source": "controller-auto-run",
                    },
                )
                auto_outcome = router.dispatch(auto_action)
                auto_outcome = self._postprocess_python_outcome(
                    workspace, router, auto_action, auto_outcome, required_files
                )
                workflow.observe(auto_action, auto_outcome)
                if "returncode" in auto_outcome.data:
                    last_executed_solver_digest = self._current_solver_digest(workspace)
                    last_solver_execution_failed = not auto_outcome.ok
                    if last_executed_solver_digest is not None:
                        seen_solver_digests.add(last_executed_solver_digest)
                controller_events.append((auto_action, auto_outcome))
                outcome.data["controller_auto_run"] = auto_outcome.as_observation()
                solver_dirty = False
                if not auto_outcome.ok:
                    outcome.ok = False
                    outcome.summary = (
                        f"{outcome.summary}; controller auto-run failed, so repair the solver before another run"
                    )

            # A successful execution is followed by the cheap structural gate
            # immediately. The model receives the exact missing-file/format
            # list instead of spending another turn asking for validation.
            latest_run = (
                controller_events[-1][1]
                if controller_events and controller_events[-1][0].tool == "run_python"
                else outcome
                if action.tool == "run_python"
                else None
            )
            if latest_run is not None and latest_run.ok:
                # Migration/repair units often ask for one Python artifact.
                # The model commonly repairs scratch/solve.py correctly and
                # executes it, but forgets the mechanical final copy.  Publish
                # that audited source under the declared name before the gate.
                if (
                    len(required_files) == 1
                    and required_files[0].lower().endswith(".py")
                    and required_files[0] not in workspace.inventory("output")
                    and "solve.py" in workspace.inventory("scratch")
                ):
                    publish_action = Action(
                        "write_file",
                        {
                            "path": f"output/{required_files[0]}",
                            "source": "controller-publish-solver",
                        },
                    )
                    try:
                        _, solve_path = workspace.resolve("scratch/solve.py")
                        published = workspace.write_file(
                            f"output/{required_files[0]}",
                            solve_path.read_text(encoding="utf-8"),
                            overwrite=True,
                        )
                        publish_outcome = ToolOutcome(
                            True,
                            f"published scratch/solve.py to output/{required_files[0]}",
                            {
                                "bytes": published.stat().st_size,
                                "sha256": hashlib.sha256(published.read_bytes()).hexdigest(),
                            },
                            mutated=True,
                        )
                    except (WorkspaceError, OSError, UnicodeError, ValueError) as exc:
                        publish_outcome = ToolOutcome(
                            False, f"controller publish failed: {type(exc).__name__}: {exc}"
                        )
                    workflow.observe(publish_action, publish_outcome)
                    controller_events.append((publish_action, publish_outcome))
                validation = workspace.validate_outputs(required_files)
                validation_action = Action(
                    "validate_outputs", {"source": "controller-post-run"}
                )
                validation_outcome = ToolOutcome(
                    validation.ok,
                    "output contract checks passed"
                    if validation.ok
                    else "output contract checks failed",
                    {
                        "files": list(validation.files),
                        "issues": list(validation.issues),
                        "warnings": list(validation.warnings),
                        "artifact_manifest": workspace.output_manifest(),
                    },
                )
                workflow.observe(validation_action, validation_outcome)
                controller_events.append((validation_action, validation_outcome))
                outcome.data["controller_post_run_validation"] = (
                    validation_outcome.as_observation()
                )
                if (
                    validation_outcome.ok
                    and self.config.auto_complete_valid_run
                    and not strict_verification
                    and not workflow.state.tests_failed
                    and not workflow.state.executions_failed
                ):
                    trajectory.action(step, action, outcome)
                    for controller_action, controller_outcome in controller_events:
                        trajectory.action(step, controller_action, controller_outcome)
                    trajectory.finish("completed", validation.files)
                    return AgentResult(
                        status="completed",
                        steps=step,
                        model_calls=model_calls,
                        output_validation=validation,
                        message=(
                            "controller auto-completed after the current solver executed "
                            "and deterministic output checks passed"
                        ),
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        starter_used=starter is not None or operator_program is not None,
                        routed_category=strategy.category,
                        solver_architecture=strategy.architecture,
                    )
                if not validation_outcome.ok:
                    last_solver_execution_failed = True
                    outcome.ok = False
                    outcome.summary = (
                        f"{outcome.summary}; output contract failed, so change code/artifacts before validation"
                    )

            if latest_run is not None and "returncode" in latest_run.data:
                latest_execution_evidence = {
                    key: latest_run.data[key] for key in
                    ("output", "source_context", "repair_context", "stage_evidence")
                    if key in latest_run.data
                }
                latest_execution_evidence['previous_returncode'] = latest_run.data['returncode']
                inspection_cache.clear()

            if controller_events:
                outcome.data["workflow"] = workflow.snapshot()
                outcome.data["workflow_guidance"] = workflow.guidance()
            if action.tool in {"write_file", "replace_text", "replace_lines", "copy_file", "replace_function"} and not outcome.ok:
                failed_edit_fingerprints[failed_edit_key] = failed_edit_fingerprints.get(failed_edit_key, 0) + 1

            failed_outcomes = [outcome, *(item[1] for item in controller_events)]
            for failed in failed_outcomes:
                if failed.ok:
                    continue
                record: dict[str, object] = {
                    "kind": str(failed.data.get("workflow", {}).get("last_failure_kind", "")),
                    "summary": failed.summary[:400],
                }
                for key in ("error", "source_context", "required_next_step", "issues"):
                    value = failed.data.get(key)
                    if value:
                        record[key] = str(value)[:2400]
                output = str(failed.data.get("output", ""))
                if output:
                    record["output_tail"] = output[-2400:]
                if record not in failure_memory:
                    failure_memory.append(record)
                    failure_memory[:] = failure_memory[-12:]

            trajectory.action(step, action, outcome)
            for controller_action, controller_outcome in controller_events:
                trajectory.action(step, controller_action, controller_outcome)
            if self.config.context_memory_enabled:
                digest = self._current_solver_digest(workspace)
                # Nested auto-run failures belong to the execution gate, not
                # the successful source mutation that caused that execution.
                primary = (ToolOutcome(True, "source mutation applied", mutated=True)
                           if controller_events and outcome.mutated else outcome)
                memory.observe(step, action, primary, digest)
                for controller_action, controller_outcome in controller_events:
                    memory.observe(step, controller_action, controller_outcome, digest)
                memory.save(memory.checkpoint(
                    workflow=workflow.snapshot(), required_files=required_files,
                    solver_digest=digest, budget={"model_calls_used": model_calls},
                ))
            action_history.append(
                {
                    "step": step,
                    "tool": action.tool,
                    "ok": outcome.ok,
                    "summary": outcome.summary[:240],
                    "failure_kind": str(outcome.data.get("workflow", {}).get("last_failure_kind", "")),
                }
            )
            messages.append({"role": "assistant", "content": compact_action(action)})
            messages.append({"role": "user", "content": (
                context.observation(outcome) if self.config.context_memory_enabled
                else self._observation(outcome)
            )})
            if actions_without_mutation.get(fingerprint, 0) > self.config.max_identical_actions:
                stop_reason = "stalled: repeated unchanged action after evidence replay; no further model calls"
                break

        # A common small-model failure mode is spending its final turn repairing
        # solve.py. Execute that fresh code once without another House request so
        # a successful last edit is not discarded merely because the model-call
        # budget ended.
        remaining = deadline - time.monotonic() - self.config.reserve_sec
        if solver_dirty and remaining > 1.0 and "solve.py" in workspace.inventory("scratch"):
            rescue_action = Action(
                "run_python",
                {
                    "script": "scratch/solve.py",
                    "args": [],
                    "timeout_sec": min(120.0, remaining),
                    "source": "controller-final-rescue",
                },
            )
            try:
                rescue_outcome = workspace.run_python(
                    "scratch/solve.py", [], min(120.0, remaining)
                )
            except (WorkspaceError, OSError, UnicodeError, ValueError) as exc:
                rescue_outcome = ToolOutcome(False, f"{type(exc).__name__}: {exc}")
            workflow.observe(rescue_action, rescue_outcome)
            if rescue_outcome.ok:
                rescue_outcome.data["artifact_manifest"] = workspace.output_manifest()
            trajectory.action(effective_steps + 1, rescue_action, rescue_outcome)

        validation = workspace.validate_outputs(required_files)
        if strict_verification:
            workflow.state.validation_passed = validation.ok
            verification_block = workflow.guard(Action('finish'))
            if verification_block is not None and not stop_reason:
                stop_reason = verification_block.summary
        status = "completed" if (validation.ok and not workflow.state.tests_failed
                                 and not workflow.state.executions_failed and not stop_reason) else "incomplete"
        trajectory.finish(status, validation.files)
        return AgentResult(
            status=status,
            steps=min(effective_steps, model_calls + format_errors),
            model_calls=model_calls,
            output_validation=validation,
            message=(
                stop_reason or "budget ended with generically valid deliverables"
                if status == "completed" or stop_reason
                else "self-authored tests still failing"
                if workflow.state.tests_failed
                else "script execution still failing"
                if workflow.state.executions_failed
                else "budget ended before valid deliverables were produced"
            ),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            starter_used=starter is not None or operator_program is not None,
            routed_category=strategy.category,
            solver_architecture=strategy.architecture,
        )
