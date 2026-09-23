from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qfa_agent.agent import AgentConfig, CodingAgent
from qfa_agent.context import ContextBudgetError, ContextManager, estimate_tokens
from qfa_agent.memory import WorkingMemory
from qfa_agent.model import ModelReply
from qfa_agent.strategies import STRATEGIES
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import WorkflowController
from qfa_agent.workspace import TaskWorkspace


class ContextMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for folder in ('input', 'output', 'scratch'):
            (self.root / folder).mkdir()
        self.workspace = TaskWorkspace(*(self.root / name for name in ('input', 'output', 'scratch')),
                                       canaries=('private-canary',))

    def test_large_observation_offloads_and_retains_traceback_tail(self):
        context = ContextManager(self.workspace)
        outcome = ToolOutcome(False, 'execution failed', {
            'output': 'x' * 80000 + '\nNameError: missing_rate',
            'source_context': '18: price = missing_rate', 'error': 'private-canary',
        })
        visible = json.loads(context.observation(outcome))
        self.assertLess(len(json.dumps(visible)), 6000)
        self.assertIn('NameError', visible['data_preview']['output'])
        _, stored = self.workspace.resolve(visible['evidence']['path'])
        self.assertNotIn('private-canary', stored.read_text())
        saved = json.loads(stored.read_text())
        self.assertIn('x' * 10000, ''.join(saved['data']['output']['chunks']))
        self.assertGreater(len(stored.read_text().splitlines()), 70)
        self.assertTrue(visible['evidence']['complete'])

    def test_extreme_observation_has_explicit_storage_truncation(self):
        context = ContextManager(self.workspace)
        payload = json.loads(context.observation(ToolOutcome(False, 'huge output',
                                                             {'output': 'x' * 2_000_000})))
        self.assertFalse(payload['evidence']['complete'])
        _, stored = self.workspace.resolve(payload['evidence']['path'])
        self.assertLess(stored.stat().st_size, 500000)
        self.assertTrue(json.loads(stored.read_text())['truncated'])

    def test_context_budget_keeps_instruction_and_whole_recent_pair(self):
        context = ContextManager(self.workspace, max_tokens=1400)
        messages = [{'role': 'system', 'content': 'tools'},
                    {'role': 'user', 'content': 'Exact output contract'}]
        for index in range(10):
            messages.extend([{'role': 'assistant', 'content': f'action {index}'},
                             {'role': 'user', 'content': '测' * 500}])
        packed, stats = context.build(messages, {'active_failures': ['preserve KeyError']})
        self.assertEqual(packed[:2], messages[:2])
        self.assertEqual(packed[-2:], messages[-2:])
        self.assertEqual((len(packed) - 3) % 2, 0)
        self.assertLessEqual(estimate_tokens(packed), 1400)
        self.assertGreater(stats['dropped_history_messages'], 0)
        self.assertIn('KeyError', packed[2]['content'])

    def test_pinned_overflow_is_explicit(self):
        with self.assertRaises(ContextBudgetError):
            ContextManager(self.workspace, max_tokens=100).build([
                {'role': 'system', 'content': 'rules' * 1000},
                {'role': 'user', 'content': 'task'},
            ], {})

    def test_memory_preserves_failure_until_matching_gate_passes(self):
        memory = WorkingMemory(self.workspace, 'task-A', 'risk-management')
        run = Action('run_python', {'script': 'scratch/solve.py'})
        check = Action('validate_outputs', {})
        memory.observe(1, run, ToolOutcome(False, 'ValueError: bad units'), 'old')
        memory.observe(2, check, ToolOutcome(False, 'missing output'), 'old')
        memory.observe(3, Action('run_python', {'script': 'scratch/other.py'}),
                       ToolOutcome(True, 'ok'), 'new')
        self.assertEqual(sum(f['status'] == 'active' for f in memory.failures.values()), 2)
        memory.observe(4, run, ToolOutcome(True, 'ok'), 'new')
        snapshot = memory.checkpoint(workflow={}, required_files=('x.json',),
                                     solver_digest='new', budget={})
        self.assertEqual(len(snapshot['active_failures']), 1)
        self.assertEqual(snapshot['active_failures'][0]['tool'], 'validate_outputs')
        self.assertTrue(snapshot['active_failures'][0]['source_changed_since_failure'])
        self.assertEqual(snapshot['resolved_failures'][0]['verified_digest'], 'new')
        memory.save(snapshot)
        saved = json.loads((self.root / 'scratch/.agent/memory.json').read_text())
        self.assertEqual(saved['task_key'], 'task-A')
        self.assertFalse(saved['automatic_resume'])
        fresh = WorkingMemory(self.workspace, 'task-B', 'credit')
        self.assertEqual(fresh.failures, {})

    def test_repeated_failure_is_deduplicated_and_recipe_is_retrieved(self):
        memory = WorkingMemory(self.workspace, 'task-A', 'risk-management')
        for step in (1, 2):
            memory.observe(step, Action('run_python', {'script': 'scratch/solve.py'}),
                           ToolOutcome(False, 'KeyError: price'), 'hash')
        snapshot = memory.checkpoint(workflow={}, required_files=(), solver_digest='hash', budget={})
        self.assertEqual(len(snapshot['active_failures']), 1)
        self.assertEqual(snapshot['active_failures'][0]['count'], 2)
        self.assertEqual(snapshot['procedural_memory'][0]['kind'], 'schema')

    def test_schema_validation_cannot_clear_failed_or_stale_tests(self):
        workflow = WorkflowController(STRATEGIES['risk-management'])
        original = Action('run_pytest', {'paths': ['scratch/test_original.py']})
        workflow.observe(original, ToolOutcome(False, 'assertionerror'))
        workflow.observe(Action('validate_outputs', {}), ToolOutcome(True, 'valid files'))
        workflow.observe(Action('run_pytest', {'paths': ['scratch/test_other.py']}),
                         ToolOutcome(True, 'passed'))
        self.assertIsNotNone(workflow.guard(Action('finish')))
        workflow.observe(original, ToolOutcome(True, 'passed'))
        self.assertIsNone(workflow.guard(Action('finish')))
        workflow.observe(Action('write_file', {'path': 'scratch/solve.py'}),
                         ToolOutcome(True, 'changed', mutated=True))
        workflow.observe(Action('validate_outputs'), ToolOutcome(True, 'valid files'))
        self.assertIsNotNone(workflow.guard(Action('finish')))

    def test_agent_retrieves_repair_memory_after_failed_execution(self):
        task_root = self.root / 'input'
        (task_root / 'instruction.md').write_text('Write output/results.json.')
        (task_root / 'card.toml').write_text(
            'schema_version="2.0"\n[task]\nid="memory-integration"\n[agent]\ntimeout_sec=60\n')
        observed = []
        source = "import os\nfrom pathlib import Path\n(Path(os.environ['OUTPUT_DIR'])/'results.json').write_text('{\"value\": 1}')\n"

        class InspectingModel:
            name = 'replay'

            def complete(self, messages, *, timeout_sec):
                observed.append(messages)
                actions = [
                    {'tool': 'write_file', 'arguments': {'path': 'scratch/solve.py', 'content': "raise ValueError('units')"}},
                    {'tool': 'write_file', 'arguments': {'path': 'scratch/solve.py', 'content': source, 'overwrite': True}},
                    {'tool': 'finish', 'arguments': {}},
                ]
                return ModelReply(actions[len(observed) - 1])

        with patch('qfa_agent.agent.trusted_operator_program_for') as shortcut:
            result = CodingAgent(InspectingModel(), AgentConfig(
                starters_enabled=False, max_steps=3, reserve_sec=0,
            )).solve(load_task(task_root), self.workspace, Trajectory(self.root / 'trace.jsonl'))
            shortcut.assert_not_called()
        self.assertTrue(result.succeeded)
        second = json.loads(observed[1][2]['content'])
        third = json.loads(observed[2][2]['content'])
        self.assertTrue(any(f['tool'] == 'run_python' for f in second['active_failures']))
        self.assertTrue(any(f['tool'] == 'run_python' for f in third['resolved_failures']))
        self.assertFalse(any(f['tool'] == 'run_python' for f in third['active_failures']))
        events = [json.loads(line) for line in (self.root / 'trace.jsonl').read_text().splitlines()]
        self.assertEqual(sum(e['event'] == 'context' for e in events), 3)

    def test_budget_exhaustion_cannot_pass_with_failed_self_tests(self):
        task_root = self.root / 'input'
        (task_root / 'instruction.md').write_text('Write output/results.json.')
        (task_root / 'card.toml').write_text(
            'schema_version="2.0"\n[task]\nid="failed-audit"\n[agent]\ntimeout_sec=60\n')
        actions = iter([
            {'tool': 'write_file', 'arguments': {'path': 'output/results.json', 'content': '{"value": 1}'}},
            {'tool': 'write_file', 'arguments': {'path': 'scratch/test_solution.py',
                                               'content': 'def test_units():\n    assert False\n'}},
            {'tool': 'run_pytest', 'arguments': {'paths': ['scratch/test_solution.py']}},
        ])

        class Model:
            name = 'replay'

            def complete(self, messages, *, timeout_sec):
                return ModelReply(next(actions))

        result = CodingAgent(Model(), AgentConfig(starters_enabled=False, max_steps=3, reserve_sec=0)).solve(
            load_task(task_root), self.workspace, Trajectory(self.root / 'trace.jsonl'))
        self.assertTrue(result.output_validation.ok)
        self.assertFalse(result.succeeded)
        self.assertIn('tests still failing', result.message)


if __name__ == '__main__':
    unittest.main()
