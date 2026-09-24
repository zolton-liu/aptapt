from __future__ import annotations

import json
import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from qfa_agent.agent import AgentConfig, CodingAgent
from qfa_agent.model import ModelError, ReplayModel
from qfa_agent.protocol import Action
from qfa_agent.starters import StarterProgram
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.workspace import TaskWorkspace


class AgentIntegrationTests(unittest.TestCase):
    def test_local_operator_adapter_gate_accepts_only_one_short_top_level_call(self) -> None:
        operator = "write_variance_swap_outputs"
        valid = (
            "from qfa_agent.finance_ops import write_variance_swap_outputs\n"
            "write_variance_swap_outputs('chain.csv', 'params.json', 'output')\n"
        )
        with patch.dict(os.environ, {"QFA_STAGE_PIPELINE": "0"}):
            self.assertIsNone(
                CodingAgent._required_operator_adapter_issue(valid, operator)
            )
            self.assertIn(
                "exactly once",
                CodingAgent._required_operator_adapter_issue(
                    valid + "write_variance_swap_outputs('a', 'b', 'c')\n",
                    operator,
                ),
            )
            self.assertIn(
                "no functions or loops",
                CodingAgent._required_operator_adapter_issue(
                    "from qfa_agent.finance_ops import write_variance_swap_outputs\n"
                    "for _ in range(1):\n"
                    "    write_variance_swap_outputs('a', 'b', 'c')\n",
                    operator,
                ),
            )

    def test_strict_mode_allows_staged_required_operator_wrapper(self) -> None:
        source = (
            "def compute(inputs):\n"
            "    return write_american_option_fd_outputs(inputs['candidate_dir'])\n"
        )
        with patch.dict(os.environ, {"QFA_STAGE_PIPELINE": "1"}):
            self.assertIsNone(
                CodingAgent._required_operator_adapter_issue(
                    source, "write_american_option_fd_outputs"
                )
            )

    def test_operator_adapter_gate_previews_followup_line_edits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root, output, scratch = root / "task", root / "output", root / "scratch"
            task_root.mkdir(); output.mkdir(); scratch.mkdir()
            workspace = TaskWorkspace(task_root, output, scratch)
            workspace.write_file(
                "scratch/solve.py",
                "from qfa_agent.finance_ops import write_variance_swap_outputs\n"
                "write_variance_swap_outputs('a', 'b', 'c')\n",
            )
            action = Action(
                "replace_lines",
                {
                    "path": "scratch/solve.py",
                    "start_line": 2,
                    "end_line": 2,
                    "content": "for _ in range(2):\n"
                    "    write_variance_swap_outputs('a', 'b', 'c')",
                },
            )
            candidate = CodingAgent._candidate_solver_source(workspace, action)
            self.assertIsNotNone(candidate)
            with patch.dict(os.environ, {"QFA_STAGE_PIPELINE": "0"}):
                self.assertIn(
                    "no functions or loops",
                    CodingAgent._required_operator_adapter_issue(
                        candidate or "", "write_variance_swap_outputs"
                    ),
                )

    def test_local_compatibility_mode_auto_completes_after_valid_execution(self) -> None:
        class OneReplyModel:
            name = "one-reply"

            def __init__(self, action):
                self.action = action
                self.calls = 0

            def complete(self, messages, *, timeout_sec):
                del messages, timeout_sec
                self.calls += 1
                if self.calls > 1:
                    raise AssertionError("a valid current execution should not need a finish turn")
                from qfa_agent.model import ModelReply
                return ModelReply(self.action)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root, output, scratch = root / "task", root / "output", root / "scratch"
            task_root.mkdir(); output.mkdir(); scratch.mkdir()
            (task_root / "instruction.md").write_text("Write output/results.json.")
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="local-auto-complete"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n'
            )
            source = (
                "import json, os\nfrom pathlib import Path\n"
                "(Path(os.environ['OUTPUT_DIR'])/'results.json').write_text(json.dumps({'value': 1}))\n"
            )
            model = OneReplyModel({"tool": "write_file", "arguments": {
                "path": "scratch/solve.py", "content": source}})
            with patch.dict(os.environ, {"QFA_STAGE_PIPELINE": "0",
                                          "QFA_VERIFICATION_REQUIRED": "0"}):
                result = CodingAgent(model, AgentConfig(
                    max_steps=3, reserve_sec=0, auto_complete_valid_run=True
                )).solve(load_task(task_root), TaskWorkspace(task_root, output, scratch),
                        Trajectory(scratch / "trace.jsonl"))
            self.assertTrue(result.succeeded)
            self.assertEqual(result.model_calls, 1)
            self.assertIn("auto-completed", result.message)

    def test_controller_auto_runs_solver_and_validates_before_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Write output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="auto-run"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            source = (
                "import json, os\nfrom pathlib import Path\n"
                "(Path(os.environ['OUTPUT_DIR']) / 'results.json').write_text(json.dumps({'value': 1}))\n"
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps({"responses": [
                    {"tool": "write_file", "arguments": {"path": "scratch/solve.py", "content": source}},
                    {"tool": "finish", "arguments": {"summary": "done"}},
                ]}),
                encoding="utf-8",
            )
            result = CodingAgent(
                ReplayModel(replay), AgentConfig(max_steps=2, reserve_sec=0)
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(scratch / "trace.jsonl"),
            )
            self.assertTrue(result.succeeded)
            self.assertEqual(result.model_calls, 2)

    def test_transient_model_error_is_retried_inside_solve_loop(self) -> None:
        class FlakyModel:
            name = "flaky"

            def __init__(self):
                self.calls = 0

            def complete(self, messages, *, timeout_sec):
                del messages, timeout_sec
                self.calls += 1
                if self.calls == 1:
                    raise ModelError("temporary disconnect")
                responses = [
                    {"tool": "write_file", "arguments": {"path": "output/results.json", "content": '{"value": 1}'}},
                    {"tool": "validate_outputs", "arguments": {}},
                    {"tool": "finish", "arguments": {"summary": "done"}},
                ]
                from qfa_agent.model import ModelReply

                return ModelReply(responses[self.calls - 2])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Write output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="flaky-model"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            result = CodingAgent(
                FlakyModel(),
                AgentConfig(max_steps=4, max_model_errors=2, reserve_sec=0),
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(scratch / "trace.jsonl"),
            )
            self.assertTrue(result.succeeded)
            self.assertEqual(result.model_calls, 4)

    def test_solver_source_oscillation_is_blocked_before_another_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Write output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="cycle"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            source_a = "raise ValueError('A')\n"
            replay = root / "replay.json"
            replay.write_text(
                json.dumps({"responses": [
                    {"tool": "write_file", "arguments": {"path": "scratch/solve.py", "content": source_a}},
                    {"tool": "replace_text", "arguments": {"path": "scratch/solve.py", "old": "'A'", "new": "'B'"}},
                    {"tool": "replace_text", "arguments": {"path": "scratch/solve.py", "old": "'B'", "new": "'A'"}},
                ]}),
                encoding="utf-8",
            )
            trace = scratch / "trace.jsonl"
            CodingAgent(
                ReplayModel(replay), AgentConfig(max_steps=3, reserve_sec=0)
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(trace),
            )
            self.assertIn("'B'", (scratch / "solve.py").read_text())
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            auto_runs = [
                event
                for event in events
                if event.get("event") == "tool" and event.get("tool") == "run_python"
            ]
            self.assertEqual(len(auto_runs), 2)
            self.assertTrue(
                any("previously executed" in event.get("summary", "") for event in events)
            )

    def test_unchanged_failed_solver_is_not_executed_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Write output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="unchanged-run"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps({"responses": [
                    {"tool": "write_file", "arguments": {"path": "scratch/solve.py", "content": "raise ValueError('broken')\n"}},
                    {"tool": "run_python", "arguments": {"script": "scratch/solve.py"}},
                    {"tool": "write_file", "arguments": {"path": "output/results.json", "content": '{"value": 1}'}},
                    {"tool": "validate_outputs", "arguments": {}},
                    {"tool": "finish", "arguments": {"summary": "done"}},
                ]}),
                encoding="utf-8",
            )
            trace = scratch / "trace.jsonl"
            result = CodingAgent(
                ReplayModel(replay), AgentConfig(max_steps=5, reserve_sec=0)
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(trace),
            )
            # A manual artifact write cannot hide the still-failing solver.
            self.assertFalse(result.succeeded)
            self.assertIn("execution still failing", result.message)
            events = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertTrue(
                any("unchanged solve.py" in event.get("summary", "") for event in events)
            )

    def test_fragile_task_family_rejects_reimplemented_solver(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "CIR short-rate calibration. Write output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="operator-gate"\n'
                '[metadata]\ncategory="fixed-income"\ndifficulty="medium"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps({"responses": [
                    {"tool": "write_file", "arguments": {
                        "path": "scratch/solve.py", "content": "print('manual CIR')\n"
                    }},
                    {"tool": "write_file", "arguments": {
                        "path": "output/results.json", "content": '{"value": 1}'
                    }},
                    {"tool": "validate_outputs", "arguments": {}},
                    {"tool": "finish", "arguments": {"summary": "done"}},
                ]}),
                encoding="utf-8",
            )
            result = CodingAgent(
                ReplayModel(replay), AgentConfig(max_steps=4, reserve_sec=0)
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(scratch / "trace.jsonl"),
            )
            self.assertTrue(result.succeeded)
            self.assertFalse((scratch / "solve.py").exists())

    def test_missing_script_failure_remains_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text("Write output/results.json.", encoding="utf-8")
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="recover-missing"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps(
                    {"responses": [
                        {"tool": "run_python", "arguments": {"script": "scratch/solve.py"}},
                        {"tool": "write_file", "arguments": {"path": "output/results.json", "content": '{"value": 1}'}},
                        {"tool": "validate_outputs", "arguments": {}},
                        {"tool": "finish", "arguments": {"summary": "done"}},
                    ]}
                ),
                encoding="utf-8",
            )
            result = CodingAgent(
                ReplayModel(replay), AgentConfig(max_steps=4, reserve_sec=0)
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(scratch / "trace.jsonl"),
            )
            self.assertTrue(result.succeeded)

    def test_final_dirty_solver_is_executed_without_another_model_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text("Write output/results.json.", encoding="utf-8")
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="final-rescue"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            replay = root / "replay.json"
            source = (
                "import json, os\nfrom pathlib import Path\n"
                "(Path(os.environ['OUTPUT_DIR']) / 'results.json').write_text(json.dumps({'value': 1}))\n"
            )
            replay.write_text(
                json.dumps({"responses": [{"tool": "write_file", "arguments": {"path": "scratch/solve.py", "content": source}}]}),
                encoding="utf-8",
            )
            result = CodingAgent(
                ReplayModel(replay), AgentConfig(max_steps=1, reserve_sec=0)
            ).solve(
                load_task(task_root),
                TaskWorkspace(task_root, output, scratch),
                Trajectory(scratch / "trace.jsonl"),
            )
            self.assertTrue(result.succeeded)
            self.assertEqual(result.model_calls, 1)

    def test_environment_can_disable_category_starters(self) -> None:
        with patch.dict(os.environ, {"QFA_DISABLE_STARTERS": "true"}, clear=False):
            config = AgentConfig.from_environment()
        self.assertFalse(config.starters_enabled)

    def test_successful_category_starter_skips_model_call(self) -> None:
        class UnexpectedModel:
            name = "must-not-run"

            def complete(self, messages, *, timeout_sec):
                del messages, timeout_sec
                raise AssertionError("model should not be called after a valid starter")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Create output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="starter-test"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            starter = StarterProgram(
                path="scratch/solve.py",
                content=(
                    "import json, os\n"
                    "from pathlib import Path\n"
                    "target = Path(os.environ['OUTPUT_DIR']) / 'results.json'\n"
                    "target.write_text(json.dumps({'value': 1.0}))\n"
                ),
                purpose="test fixture",
            )
            task = load_task(task_root)
            workspace = TaskWorkspace(task_root, output, scratch)
            with patch("qfa_agent.agent.starter_for", return_value=starter):
                result = CodingAgent(
                    UnexpectedModel(), AgentConfig(max_steps=5, reserve_sec=0)
                ).solve(task, workspace, Trajectory(scratch / "trace.jsonl"))

            self.assertTrue(result.succeeded)
            self.assertEqual(result.model_calls, 0)
            self.assertEqual(result.steps, 0)

    def test_successful_trusted_operator_skips_slow_model(self) -> None:
        class UnexpectedModel:
            name = "must-not-run"

            def complete(self, messages, *, timeout_sec):
                del messages, timeout_sec
                raise AssertionError("model should not run after trusted operator succeeds")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Create output/results.json.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="operator-test"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            program = StarterProgram(
                path="scratch/solve.py",
                content=(
                    "import json, os\nfrom pathlib import Path\n"
                    "target = Path(os.environ['OUTPUT_DIR']) / 'results.json'\n"
                    "target.write_text(json.dumps({'value': 1.0}))\n"
                ),
                purpose="trusted test operator",
            )
            task = load_task(task_root)
            workspace = TaskWorkspace(task_root, output, scratch)
            with patch(
                "qfa_agent.agent.trusted_operator_program_for", return_value=program
            ):
                result = CodingAgent(
                    UnexpectedModel(),
                    AgentConfig(max_steps=5, reserve_sec=0, starters_enabled=True),
                ).solve(task, workspace, Trajectory(scratch / "trace.jsonl"))

            self.assertTrue(result.succeeded)
            self.assertEqual(result.model_calls, 0)
            self.assertTrue(result.starter_used)

    def test_replay_model_completes_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_root = root / "task"
            output = root / "output"
            scratch = root / "scratch"
            task_root.mkdir()
            output.mkdir()
            scratch.mkdir()
            (task_root / "instruction.md").write_text(
                "Write output/results.json containing a finite value.", encoding="utf-8"
            )
            (task_root / "card.toml").write_text(
                'schema_version="2.0"\n[task]\nid="tiny"\n'
                '[provenance]\ndata_cutoff=""\n[contamination]\ncanary_guid=""\n'
                '[agent]\ntimeout_sec=30\n',
                encoding="utf-8",
            )
            replay = root / "replay.json"
            replay.write_text(
                json.dumps(
                    {
                        "responses": [
                            {
                                "tool": "write_file",
                                "arguments": {
                                    "path": "output/results.json",
                                    "content": '{"value": 1.0}\n',
                                },
                            },
                            {"tool": "validate_outputs", "arguments": {}},
                            {"tool": "finish", "arguments": {"summary": "done"}},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            task = load_task(task_root)
            workspace = TaskWorkspace(task_root, output, scratch)
            result = CodingAgent(ReplayModel(replay), AgentConfig(max_steps=5, reserve_sec=0)).solve(
                task, workspace, Trajectory(scratch / "trace.jsonl")
            )
            self.assertTrue(result.succeeded)
            self.assertEqual(json.loads((output / "results.json").read_text())["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
