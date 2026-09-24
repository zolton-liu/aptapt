from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qfa_agent.agent import CodingAgent, AgentConfig
from qfa_agent.contracts import (ContractError, input_path, json_object, require_aligned,
                                 require_columns, require_finite, require_correlation,
                                 require_price_bounds, require_accounting)
from qfa_agent.memory import WorkingMemory
from qfa_agent.repair import repair_context
from qfa_agent.stages import run_stages
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workspace import TaskWorkspace, ToolRouter
from qfa_agent.context import ContextManager
from qfa_agent.model import ModelReply
from qfa_agent.strategies import STRATEGIES
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.workflow import WorkflowController, classify_failure


class ContractPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('input', 'output', 'scratch'):
            (self.root / name).mkdir()
        self.workspace = TaskWorkspace(*(self.root / name for name in ('input', 'output', 'scratch')))
        data = self.root / 'input/environment/data'
        data.mkdir(parents=True)
        (data / 'values.csv').write_text('date,value\n2020-01-01,1\n2020-01-02,2\n')
        (data / 'config.json').write_text('{"filters": 0.25, "nested": {"enabled": true}}')
        self.env = patch.dict(os.environ, {'TASK_DIR': str(self.root / 'input'),
                                         'OUTPUT_DIR': str(self.root / 'output'),
                                         'QFA_STAGE_PIPELINE': '1', 'QFA_REPAIR_CONTEXT': '1',
                                         'QFA_VERSIONED_MEMORY': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_exact_input_path_no_guessing_or_hidden_material(self):
        self.assertTrue(input_path('environment/data/values.csv').is_file())
        for name in ('values.csv', '../output/x', '/etc/passwd', 'checks/test_outputs.py'):
            with self.subTest(name=name), self.assertRaises(ContractError):
                input_path(name)
        hidden = self.root / 'input/checks'
        hidden.mkdir()
        (hidden / 'test_outputs.py').write_text('secret')
        with self.assertRaises(ContractError):
            input_path('checks/test_outputs.py')

    def test_json_scalar_type_is_not_silently_changed_to_object(self):
        data = json_object('environment/data/config.json', fields={('filters',): float})
        self.assertEqual(data['filters'], 0.25)
        for fields in ({('filters',): dict}, {('filters', 'max_relative_spread'): float},
                       {('nested', 'enabled'): int}):
            with self.assertRaises(ContractError):
                json_object('environment/data/config.json', fields=fields)

    def test_alignment_checks_labels_not_just_length_and_never_mutates(self):
        a = pd.Series([1, 2], index=['a', 'b'])
        b = pd.DataFrame({'x': [2, 3]}, index=['b', 'a'])
        with self.assertRaisesRegex(ContractError, 'alignment mismatch'):
            require_aligned(a, b)
        self.assertEqual(list(b.index), ['b', 'a'])
        require_aligned(a, b.reindex(a.index))
        with self.assertRaises(ContractError):
            require_aligned(a, b.iloc[:1])

    def test_columns_and_finiteness_do_not_clean_or_fill_bad_values(self):
        frame = pd.DataFrame({'x': [1, float('nan')]})
        require_columns(frame, ['x'])
        with self.assertRaises(ContractError):
            require_columns(frame, ['y'])
        with self.assertRaisesRegex(ContractError, 'non-finite'):
            require_finite(frame.x, label='return')
        self.assertTrue(pd.isna(frame.x.iloc[1]))

    def test_domain_checks_have_distinct_executable_invariants(self):
        require_correlation([[1, 0.2], [0.2, 1]])
        with self.assertRaises(ContractError):
            require_correlation([[1, 0.9, 0.9], [0.9, 1, -0.9], [0.9, -0.9, 1]])
        require_price_bounds([2, 3], 0, [5, 5])
        with self.assertRaises(ContractError):
            require_price_bounds([2, -1], 0, [5, 5])
        with self.assertRaises(ContractError):
            require_price_bounds([2, 3], [[0], [0]], 5)
        require_accounting([100, 105], [20, 25], [80, 80])
        with self.assertRaises(ContractError):
            require_accounting([100, 105], [20, 25], [80, 79])
        with self.assertRaises(ContractError):
            require_price_bounds([-1], 0, 5, atol=float('inf'))

    def test_numeric_and_alignment_errors_not_misclassified_as_generic_types(self):
        self.assertEqual(classify_failure(ToolOutcome(False, 'ValueError: non-finite nan')), 'numeric')
        self.assertEqual(classify_failure(ToolOutcome(False, 'ValueError: operands could not be broadcast together')), 'alignment')

    def test_mutation_invalidates_stage_evidence(self):
        workflow = WorkflowController(STRATEGIES['factor-research'])
        workflow.observe(Action('run_python', {'script': 'scratch/solve.py'}),
                         ToolOutcome(True, 'ok', {'returncode': 0, 'stage_evidence': {
                             'mode': 'staged', 'complete': True, 'stages': []}}))
        self.assertTrue(workflow.snapshot()['solver_stages']['complete'])
        workflow.observe(Action('replace_text', {'path': 'scratch/solve.py'}),
                         ToolOutcome(True, 'changed', mutated=True))
        self.assertFalse(workflow.snapshot()['solver_stages']['complete'])
        self.assertTrue(workflow.snapshot()['solver_stages']['stale'])

    def test_new_repair_evidence_is_bounded_when_offloaded(self):
        packet = {'path': 'scratch/solve.py', 'line': 3, 'location_verified': True,
                  'source_context': 'x' * 5000, 'library_tail': 'y' * 1600}
        raw = ToolOutcome(False, 'failed', {'output': 'z' * 80000, 'repair_context': packet})
        visible = ContextManager(self.workspace).observation(raw)
        self.assertLess(len(visible), 6000)
        self.assertEqual(json.loads(visible)['data_preview']['repair_context']['line'], 3)

    def test_traceback_uses_owned_frame_and_all_variable_references(self):
        source = "data_path = task_dir / 'values.csv'\nother = 3\ndata = pd.read_csv(data_path)\nwrite_operator(data_path)\n"
        self.workspace.write_file('scratch/solve.py', source)
        path = self.root / 'scratch/solve.py'
        output = f'  File "{path}", line 3, in <module>\n  File "/libraries/pandas/io.py", line 873, in open\nFileNotFoundError: [Errno 2] No such file or directory: \'{self.root}/input/values.csv\''
        packet = repair_context(self.workspace, output)
        self.assertEqual(packet['line'], 3)
        self.assertEqual(packet['path'], 'scratch/solve.py')
        self.assertIn('1: data_path', packet['source_context'])
        self.assertIn('4: write_operator(data_path)', packet['source_context'])
        self.assertEqual(packet['visible_path_candidates'], ['input/environment/data/values.csv'])
        outcome = CodingAgent._postprocess_python_outcome(
            self.workspace, ToolRouter(self.workspace), Action('run_python', {'script': 'scratch/solve.py'}),
            ToolOutcome(False, 'failed', {'output': output, 'returncode': 1}), ())
        self.assertEqual(outcome.data['repair_context']['line'], 3)

    def test_helper_frame_is_not_mapped_to_solve_and_input_frames_are_hidden(self):
        self.workspace.write_file('scratch/helper.py', 'raise ValueError("bad")\n')
        out = f'File "{self.root}/scratch/helper.py", line 1\nFile "/lib/library.py", line 500\n'
        self.assertEqual(repair_context(self.workspace, out)['path'], 'scratch/helper.py')
        self.assertFalse(repair_context(self.workspace, f'File "{self.root}/input/checks/secret.py", line 1')['location_verified'])

    def test_versioned_memory_stales_changed_source_and_replays_without_new_failure(self):
        m = WorkingMemory(self.workspace, 'fixture', 'factor-research')
        run = Action('run_python', {'script': 'scratch/solve.py'})
        edit = Action('replace_lines', {'path': 'scratch/solve.py'})
        m.observe(1, run, ToolOutcome(False, 'bad shape', {'returncode': 1}), 'old')
        m.observe(2, edit, ToolOutcome(False, 'syntax rejected'), 'old')
        m.observe(3, edit, ToolOutcome(True, 'changed', mutated=True), 'new')
        self.assertTrue(all(f['status'] == 'stale' for f in m.failures.values()))
        packet = {'path': 'scratch/solve.py', 'line': 3, 'source_sha256': 'new'}
        m.observe(4, run, ToolOutcome(False, 'new traceback', {'returncode': 1, 'repair_context': packet}), 'new')
        before = len(m.failures)
        m.observe(5, run, ToolOutcome(False, 'unchanged', {'previous_returncode': 1}), 'new')
        self.assertEqual(len(m.failures), before)
        cp = m.checkpoint(workflow={}, required_files=(), solver_digest='new', budget={})
        self.assertEqual(len(cp['active_failures']), 1)
        self.assertEqual(cp['primary_repair'], packet)
        self.assertTrue(all(x['status'] != 'resolved' for x in cp['historical_failures']))

    def test_stage_audit_failure_prevents_writer(self):
        script = self.workspace.write_file('scratch/solve.py', 'QFA_STAGED = True\n')
        called = []
        ns = {'load_inputs': lambda: {'x': 2}, 'compute': lambda x: {'price': -1},
              'audit': lambda x, r: {'nonnegative_price': r['price'] >= 0},
              'write_outputs': lambda r: called.append(r)}
        evidence = self.root / 'scratch/evidence.json'
        with self.assertRaisesRegex(ValueError, 'nonnegative_price'):
            run_stages(ns, script, evidence, 'fixture')
        self.assertEqual(called, [])
        log = json.loads(evidence.read_text())
        self.assertEqual(log['stages'][-1]['name'], 'audit')
        self.assertFalse(log['complete'])

    def test_helper_edit_invalidates_its_traceback_without_changing_main_digest(self):
        m = WorkingMemory(self.workspace, 'fixture', 'factor-research')
        self.workspace.write_file('scratch/helper.py', 'raise ValueError("new")\n')
        run = Action('run_python', {'script': 'scratch/solve.py'})
        m.observe(1, run, ToolOutcome(False, 'old helper failed', {'returncode': 1,
                  'repair_context': {'path': 'scratch/helper.py', 'source_sha256': 'old-helper'}}), 'same-main')
        m.observe(2, Action('replace_text', {'path': 'scratch/helper.py'}),
                  ToolOutcome(True, 'changed', mutated=True), 'same-main')
        self.assertEqual(next(iter(m.failures.values()))['status'], 'stale')

    def test_stage_audit_rejects_truthy_strings_and_missing_checks(self):
        script = self.workspace.write_file('scratch/solve.py', 'QFA_STAGED = True\n')
        for checks in ({}, {'numeric': 'false'}, {'numeric': 1}):
            ns = {'load_inputs': lambda: {'x': 1}, 'compute': lambda x: x,
                  'audit': lambda x, r: checks, 'write_outputs': lambda r: self.fail('writer called')}
            with self.assertRaises(ValueError):
                run_stages(ns, script, self.root / 'scratch/evidence.json', 'fixture')

    def test_real_runtime_stage_failure_then_new_version_passes(self):
        source = '''QFA_STAGED = True
from qfa_agent.contracts import input_path, require_columns, require_finite
def load_inputs():
    frame = pd.read_csv(input_path('environment/data/values.csv'))
    require_columns(frame, ['date', 'value'])
    return {'frame': frame}
def compute(inputs):
    return {'sum': float(inputs['frame'].value.sum())}
def audit(inputs, result):
    require_finite(result['sum'], label='sum')
    return {'reconciles': result['sum'] == -1}
def write_outputs(result):
    (output_dir / 'result.json').write_text(json.dumps(result, allow_nan=False))
'''
        self.workspace.write_file('scratch/solve.py', source)
        first = self.workspace.run_python('scratch/solve.py', [], 20)
        self.assertFalse(first.ok)
        self.assertEqual(first.data['stage_evidence']['stages'][-1]['name'], 'audit')
        self.assertFalse((self.root / 'output/result.json').exists())
        # Restore the fixture's independent, hand-calculated expected sum;
        # never obtain an answer from a public checker.
        self.workspace.write_file('scratch/solve.py', source.replace("result['sum'] == -1", "result['sum'] == 3.0"), overwrite=True)
        second = self.workspace.run_python('scratch/solve.py', [], 20)
        self.assertTrue(second.ok, second.data['output'])
        self.assertTrue(second.data['stage_evidence']['complete'])
        self.assertNotEqual(first.data['stage_evidence']['run_id'], second.data['stage_evidence']['run_id'])
        self.assertNotEqual(first.data['stage_evidence']['source_sha256'], second.data['stage_evidence']['source_sha256'])
        self.assertEqual(json.loads((self.root / 'output/result.json').read_text()), {'sum': 3.0})

    def test_zero_exit_does_not_bypass_unfinished_stages(self):
        self.workspace.write_file('scratch/solve.py', 'QFA_STAGED = True\ndef load_inputs():\n    raise SystemExit(0)\n')
        result = self.workspace.run_python('scratch/solve.py', [], 20)
        self.assertEqual(result.data['returncode'], 0)
        self.assertFalse(result.ok)
        self.assertFalse(result.data['stage_evidence']['complete'])

    def test_legacy_script_not_mislabelled_as_staged_validation(self):
        self.workspace.write_file('scratch/solve.py', 'x = 1\n')
        result = self.workspace.run_python('scratch/solve.py', [], 20)
        self.assertTrue(result.ok)
        self.assertEqual(result.data['stage_evidence'], {'mode': 'legacy', 'complete': False})

    def test_staged_solver_integrates_with_agent_auto_run_and_finish_gate(self):
        (self.root / 'input/card.toml').write_text('[task]\nid="staged-fixture"\n[agent]\ntimeout_sec=60\n')
        (self.root / 'input/instruction.md').write_text('Sum supplied values and write output/result.json.')
        source = '''QFA_STAGED = True
from qfa_agent.contracts import input_path, require_columns
def load_inputs():
    data = pd.read_csv(input_path('environment/data/values.csv'))
    require_columns(data, ['value'])
    return {'values': data.value.tolist()}
def compute(inputs):
    return {'sum': sum(inputs['values'])}
def audit(inputs, result):
    return {'sum_equals_manual_fixture_expectation': result['sum'] == 3}
def write_outputs(result):
    (output_dir/'result.json').write_text(json.dumps(result))
'''
        class Model:
            name = 'deterministic-integration-fixture'
            def __init__(self):
                self.calls = 0
            def complete(self, messages, *, timeout_sec):
                self.calls += 1
                if self.calls == 1:
                    return ModelReply({'tool': 'write_file', 'arguments': {'path': 'scratch/solve.py', 'content': source}})
                return ModelReply({'tool': 'finish', 'arguments': {}})
        model = Model()
        result = CodingAgent(model, AgentConfig(starters_enabled=False, max_steps=2, reserve_sec=2)).solve(
            load_task(self.root / 'input'), self.workspace, Trajectory(self.root / 'trace.jsonl'))
        self.assertTrue(result.succeeded)
        self.assertEqual(model.calls, 2)
        self.assertEqual(json.loads((self.root / 'output/result.json').read_text()), {'sum': 3})
        memory = json.loads((self.root / 'scratch/.agent/memory.json').read_text())
        self.assertTrue(memory['workflow']['solver_stages']['complete'])
