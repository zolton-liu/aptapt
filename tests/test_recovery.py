"""Regression coverage for measured v6 stalls; no public task answers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qfa_agent.agent import AgentConfig, CodingAgent
from qfa_agent.context import ContextManager
from qfa_agent.memory import WorkingMemory
from qfa_agent.model import ModelReply
from qfa_agent.strategies import STRATEGIES
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import WorkflowController
from qfa_agent.workspace import TaskWorkspace, ToolRouter


class SequenceModel:
    name = 'regression-replay'

    def __init__(self, actions):
        self.actions = iter(actions)
        self.requests = []

    def complete(self, messages, *, timeout_sec):
        self.requests.append((messages, timeout_sec))
        return ModelReply(next(self.actions))


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('input', 'output', 'scratch'):
            (self.root / folder).mkdir()
        (self.root / 'input/instruction.md').write_text('Write output/results.json.')
        (self.root / 'input/card.toml').write_text('[task]\nid="recovery"\n[agent]\ntimeout_sec=2400\n')
        (self.root / 'input/data.txt').write_text('observed evidence\n')
        self.workspace = TaskWorkspace(*(self.root / p for p in ('input', 'output', 'scratch')))
        self.trace_path = self.root / 'trace.jsonl'

    def solve(self, actions, **kwargs):
        model = SequenceModel(actions)
        result = CodingAgent(model, AgentConfig(starters_enabled=False, max_steps=len(actions),
                                                reserve_sec=2)).solve(
            load_task(self.root / 'input'), self.workspace, Trajectory(self.trace_path), **kwargs)
        return result, model

    def test_outer_deadline_caps_model_request_and_card_remains_unchanged(self):
        with patch('qfa_agent.agent.time.monotonic', return_value=100):
            result, model = self.solve([
                {'tool': 'write_file', 'arguments': {'path': 'output/results.json', 'content': '{"ok":1}'}},
            ], deadline=110)
        self.assertTrue(result.succeeded)
        self.assertEqual(model.requests[0][1], 8)
        self.assertEqual(load_task(self.root / 'input').timeout_sec, 2400)

    def test_expired_deadline_does_not_call_model(self):
        with patch('qfa_agent.agent.time.monotonic', return_value=100):
            result, model = self.solve([{'tool': 'finish', 'arguments': {}}], deadline=99)
        self.assertFalse(result.succeeded)
        self.assertEqual(model.requests, [])

    def test_router_caps_execution_and_refuses_after_deadline_without_charging(self):
        self.workspace.write_file('scratch/solve.py', 'pass\n')
        router = ToolRouter(self.workspace, deadline=110)
        action = Action('run_python', {'script': 'scratch/solve.py', 'timeout_sec': 600})
        with patch('qfa_agent.workspace.time.monotonic', return_value=108), \
             patch.object(self.workspace, 'run_python', return_value=ToolOutcome(True, 'ok')) as run:
            self.assertTrue(router.dispatch(action).ok)
            self.assertEqual(run.call_args.args[2], 2)
        with patch('qfa_agent.workspace.time.monotonic', return_value=111), \
             patch.object(self.workspace, 'run_python') as run:
            self.assertFalse(router.dispatch(action).ok)
            run.assert_not_called()
        self.assertEqual(router.python_runs, 1)

    def test_repeated_read_replays_evidence_then_stops_instead_of_spending_all_calls(self):
        read = {'tool': 'read_file', 'arguments': {'path': 'input/data.txt'}}
        with patch.object(self.workspace, 'read_file', wraps=self.workspace.read_file) as reader:
            result, model = self.solve([read] * 8)
        self.assertFalse(result.succeeded)
        self.assertIn('stalled', result.message)
        self.assertEqual(len(model.requests), 3)
        matching = [call for call in reader.call_args_list if call.args[0] == 'input/data.txt']
        self.assertEqual(len(matching), 1)
        replay = json.loads(model.requests[2][0][-1]['content'])
        self.assertTrue(replay['data']['cached'])
        self.assertIn('observed evidence', replay['data']['content'])

    def test_execution_failure_cannot_be_hidden_by_partial_valid_artifact(self):
        source = "output_dir.joinpath('results.json').write_text('{\"ok\":1}')\nraise ValueError('unfinished')\n"
        result, _ = self.solve([{'tool': 'write_file', 'arguments': {
            'path': 'scratch/solve.py', 'content': source}}])
        self.assertTrue(result.output_validation.ok)
        self.assertFalse(result.succeeded)
        self.assertIn('execution still failing', result.message)

    def test_source_mutation_invalidates_cached_read(self):
        read = {'tool': 'read_file', 'arguments': {'path': 'scratch/solve.py'}}
        _, model = self.solve([
            {'tool': 'write_file', 'arguments': {'path': 'scratch/solve.py', 'content': "raise ValueError('old')\n"}},
            read,
            {'tool': 'replace_text', 'arguments': {'path': 'scratch/solve.py', 'old': "'old'", 'new': "'new'"}},
            read,
            {'tool': 'finish', 'arguments': {}},
        ])
        last_read = json.loads(model.requests[4][0][-1]['content'])
        self.assertNotIn('cached', last_read['data'])
        self.assertIn("'new'", last_read['data']['content'])

    def test_other_script_success_cannot_clear_failed_execution(self):
        workflow = WorkflowController(STRATEGIES['risk-management'])
        run = Action('run_python', {'script': 'scratch/solve.py'})
        workflow.observe(run, ToolOutcome(False, 'Python program failed', {'returncode': 1}))
        workflow.observe(Action('run_python', {'script': 'scratch/helper.py'}), ToolOutcome(True, 'ok', {'returncode': 0}))
        workflow.observe(Action('validate_outputs'), ToolOutcome(True, 'files present'))
        self.assertIsNotNone(workflow.guard(Action('finish')))
        workflow.observe(run, ToolOutcome(True, 'ok', {'returncode': 0}))
        workflow.observe(Action('validate_outputs'), ToolOutcome(True, 'files present'))
        self.assertIsNone(workflow.guard(Action('finish')))

    def test_new_failure_supersedes_old_traceback_without_claiming_resolution(self):
        memory = WorkingMemory(self.workspace, 'task', 'risk-management')
        run = Action('run_python', {'script': 'scratch/solve.py'})
        memory.observe(1, run, ToolOutcome(False, 'NameError: old'), 'old')
        memory.observe(2, run, ToolOutcome(False, 'TypeError: current'), 'new')
        checkpoint = memory.checkpoint(workflow={}, required_files=(), solver_digest='new', budget={})
        self.assertEqual(len(checkpoint['active_failures']), 1)
        self.assertIn('current', checkpoint['active_failures'][0]['summary'])
        self.assertEqual(len(checkpoint['historical_failures']), 1)
        self.assertEqual(checkpoint['resolved_failures'], [])
        memory.observe(3, run, ToolOutcome(True, 'ok'), 'fixed')
        self.assertTrue(all(f['status'] == 'resolved' for f in memory.failures.values()))

    def test_history_does_not_fill_budget_with_old_turns(self):
        messages = [{'role': 'system', 'content': 'tools'}, {'role': 'user', 'content': 'task'}]
        for index in range(20):
            messages.extend([{'role': 'assistant', 'content': f'action {index}'},
                             {'role': 'user', 'content': f'evidence {index}'}])
        packed, stats = ContextManager(self.workspace).build(messages, {})
        self.assertEqual(packed[-6:], messages[-6:])
        self.assertEqual(stats['retained_history_messages'], 6)
        self.assertEqual(stats['dropped_history_messages'], 34)

    def test_offloaded_nested_execution_keeps_actionable_error(self):
        outcome = ToolOutcome(False, 'auto-run failed', {'controller_auto_run': {
            'ok': False, 'summary': 'Python program failed',
            'data': {'output': 'x' * 10000 + '\nNameError: write_file',
                     'source_context': '35: write_file(path=output_dir / "x.json")'},
        }})
        visible = json.loads(ContextManager(self.workspace).observation(outcome))
        evidence = visible['data_preview']['controller_auto_run']
        self.assertIn('NameError: write_file', evidence['output'])
        self.assertIn('35: write_file', evidence['source_context'])

    def test_syntax_breaking_edits_preserve_last_parseable_source(self):
        original = 'def answer():\n    return 42\n'
        self.workspace.write_file('scratch/solve.py', original)
        actions = [
            Action('replace_lines', {'path': 'scratch/solve.py', 'start_line': 2,
                                     'end_line': 2, 'content': 'return 43\n'}),
            Action('replace_text', {'path': 'scratch/solve.py', 'old': '    return', 'new': 'return'}),
            Action('write_file', {'path': 'scratch/solve.py', 'content': 'def answer(:\n', 'overwrite': True}),
        ]
        router = ToolRouter(self.workspace)
        for action in actions:
            with self.subTest(tool=action.tool):
                outcome = router.dispatch(action)
                self.assertFalse(outcome.ok)
                self.assertFalse(outcome.mutated)
                self.assertIn('original source preserved', outcome.summary)
                self.assertIn('source_context', outcome.data)
                self.assertEqual((self.root / 'scratch/solve.py').read_text(), original)
        fixed = router.dispatch(Action('replace_text', {
            'path': 'scratch/solve.py', 'old': '42', 'new': '43'}))
        self.assertTrue(fixed.ok)
        self.assertIn('43', (self.root / 'scratch/solve.py').read_text())
