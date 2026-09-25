"""Regress controller defects observed in v6.3 without using task answers."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qfa_agent.agent import CodingAgent, AgentConfig
from qfa_agent.model import ModelReply
from qfa_agent.repair import repair_context
from qfa_agent.context import focused_preview
from qfa_agent.stages import run_stages
from qfa_agent.strategies import STRATEGIES
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import WorkflowController
from qfa_agent.workspace import TaskWorkspace, ToolRouter


PLAN = dict(hypothesis='unverified cause', evidence='observed execution failure',
            change='inspect and change implementation', falsification='repeat original checks')


class EvidenceProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('input', 'output', 'scratch'): (self.root/name).mkdir()
        self.workspace = TaskWorkspace(*(self.root/name for name in ('input','output','scratch')))
        self.router = ToolRouter(self.workspace)
        (self.root/'input/instruction.md').write_text('Write output/results.json from input data.')
        (self.root/'input/card.toml').write_text('[task]\nid="progress-fixture"\n[agent]\ntimeout_sec=120\n')

    def dispatch(self, tool, **args):
        return self.router.dispatch(Action(tool,args))

    def test_identical_copy_is_not_mutation_but_different_copy_is(self):
        (self.root/'input/template.py').write_text('value = 1\n')
        args = dict(source='input/template.py',destination='scratch/solve.py',overwrite=True)
        self.assertTrue(self.dispatch('copy_file',**args).mutated)
        repeat = self.dispatch('copy_file',**args)
        self.assertTrue(repeat.ok)
        self.assertFalse(repeat.mutated)
        self.assertTrue(repeat.data['no_content_change'])
        (self.root/'input/template.py').write_text('value = 2\n')
        self.assertTrue(self.dispatch('copy_file',**args).mutated)

    def test_identical_writes_and_line_edits_are_not_mutations(self):
        for content in ('value = 1\n', 'value = (\n'):
            with self.subTest(content=content):
                path='scratch/test'+str(len(content))+'.py'
                self.dispatch('write_file',path=path,content=content)
                repeated=self.dispatch('write_file',path=path,content=content,overwrite=True)
                self.assertFalse(repeated.mutated)
                self.assertFalse(self.dispatch('replace_lines',path=path,start_line=1,end_line=1,content=content).mutated)

    def test_failed_syntax_proposal_preserves_bytes(self):
        self.dispatch('write_file',path='scratch/solve.py',content='value = 1\n')
        outcome=self.dispatch('replace_lines',path='scratch/solve.py',start_line=1,end_line=1,content='value = (\n')
        self.assertFalse(outcome.ok)
        self.assertFalse(outcome.mutated)
        self.assertEqual((self.root/'scratch/solve.py').read_text(),'value = 1\n')

    def test_method_plan_idempotency_binds_source(self):
        self.assertTrue(self.dispatch('revise_method',**PLAN).mutated)
        repeat=self.dispatch('revise_method',**PLAN)
        self.assertFalse(repeat.ok)
        self.assertFalse(repeat.mutated)
        (self.root/'scratch/solve.py').write_text('value=1\n')
        self.assertTrue(self.dispatch('revise_method',**PLAN).ok)

    def test_implementation_failure_cannot_be_resolved_by_method_note(self):
        workflow=WorkflowController(STRATEGIES['cross-domain'])
        workflow.observe(Action('run_python',{'script':'scratch/solve.py'}),
                         ToolOutcome(False,'NameError: missing symbol',{'returncode':1}))
        self.assertEqual(workflow.feedback_route,'implementation')
        self.assertIsNotNone(workflow.guard(Action('revise_method',PLAN)))
        self.assertTrue(workflow.state.executions_failed)

    def test_unknown_failure_still_allows_provisional_method_review(self):
        workflow=WorkflowController(STRATEGIES['cross-domain'])
        workflow.observe(Action('run_python',{'script':'scratch/solve.py'}),
                         ToolOutcome(False,'timeout',{'returncode':1}))
        self.assertIsNone(workflow.guard(Action('revise_method',PLAN)))
        self.assertTrue(workflow.state.executions_failed)

    def test_required_operator_bootstrap_allows_only_solver_write(self):
        operator = 'write_implied_volatility_approximation_outputs'
        blocked = CodingAgent._tool_state_policy(
            Action('list_files', {'area': 'input'}),
            solver_exists=False,
            required_operator=operator,
            last_solver_execution_failed=False,
        )
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked.data['controller_policy'], 'required-operator-bootstrap')
        self.assertEqual(blocked.data['allowed_next_tools'], ['write_file'])
        allowed = CodingAgent._tool_state_policy(
            Action('write_file', {'path': 'scratch/solve.py', 'content': '# adapter'}),
            solver_exists=False,
            required_operator=operator,
            last_solver_execution_failed=False,
        )
        self.assertIsNone(allowed)

    def test_missing_solver_edit_is_build_protocol_not_repair(self):
        blocked = CodingAgent._tool_state_policy(
            Action('replace_function', {
                'path': 'scratch/solve.py', 'name': 'main', 'content': 'def main(): pass'
            }),
            solver_exists=False,
            required_operator=None,
            last_solver_execution_failed=False,
        )
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked.data['controller_policy'], 'missing-solver')
        self.assertEqual(blocked.data['policy_phase'], 'build')

    def test_failed_operator_adapter_repair_cannot_restart_exploration(self):
        blocked = CodingAgent._tool_state_policy(
            Action('copy_file', {
                'source': 'input/data/template.py', 'destination': 'scratch/solve.py'
            }),
            solver_exists=True,
            required_operator='write_fama_french_outputs',
            last_solver_execution_failed=True,
        )
        self.assertIsNotNone(blocked)
        self.assertEqual(blocked.data['controller_policy'], 'repair-evidence')
        self.assertEqual(blocked.data['policy_phase'], 'repair')

    def test_stage_signature_has_actionable_error_without_calling_bad_function(self):
        script=self.root/'scratch/solve.py'; script.write_text('# fixture\n')
        def wrong_compute(inputs,data_path):
            raise AssertionError('must not be called')
        namespace={'load_inputs':lambda:{'x':1},'compute':wrong_compute}
        evidence=self.root/'scratch/stage.json'
        with self.assertRaisesRegex(TypeError,r'stage protocol requires compute\(inputs\)'):
            run_stages(namespace,script,evidence,'fixture')
        record=json.loads(evidence.read_text())
        self.assertFalse(record['complete'])
        self.assertEqual(record['stages'][-1]['name'],'compute')
        self.assertIn('inputs mapping',record['stages'][-1]['error'])

    def test_contract_path_error_candidates_are_visible_and_preserved_in_context(self):
        for folder in ('data', 'other', 'checks'):
            (self.root/'input'/folder).mkdir()
            (self.root/'input'/folder/'params.json').write_text('{}')
        script=self.root/'scratch/solve.py'
        script.write_text('value = input_path("scratch/params.json")\n')
        output=f'File "{script}", line 1\nContractError: path does not exist: input/scratch/params.json'
        preview=focused_preview('repair_context',repair_context(self.workspace,output))
        self.assertEqual(preview['visible_path_candidates'],['input/data/params.json','input/other/params.json'])
        self.assertIn('not confirmed',preview['path_advice'])
        self.assertEqual(script.read_text(),'value = input_path("scratch/params.json")\n')

    def solve(self, actions):
        class Scripted:
            name='progress-fixture'
            def __init__(self): self.calls=0
            def complete(self,messages,*,timeout_sec):
                result=actions[self.calls]; self.calls+=1; return ModelReply(result)
        model=Scripted()
        with patch.dict(os.environ,{'QFA_VERIFICATION_REQUIRED':'0'}):
            result=CodingAgent(model,AgentConfig(starters_enabled=False,max_steps=len(actions),reserve_sec=1)).solve(
                load_task(self.root/'input'),self.workspace,Trajectory(self.root/'trace.jsonl'))
        return result,model

    def test_copy_run_note_cycle_stops_without_exhausting_eighteen_calls(self):
        (self.root/'input/template.py').write_text('raise RuntimeError("fixture failure")\n')
        cycle=[{'tool':'copy_file','arguments':dict(source='input/template.py',destination='scratch/solve.py',overwrite=True)},
               {'tool':'run_python','arguments':{'script':'scratch/solve.py'}},
               {'tool':'revise_method','arguments':PLAN}]
        result,model=self.solve(cycle*6)
        self.assertFalse(result.succeeded)
        self.assertIn('stalled',result.message)
        self.assertLess(model.calls,18)

    def test_failed_edit_can_become_valid_after_target_changes(self):
        edit={'tool':'replace_text','arguments':dict(path='scratch/helper.txt',old='second',new='third')}
        actions=[{'tool':'write_file','arguments':dict(path='scratch/helper.txt',content='first')},edit,
                 {'tool':'write_file','arguments':dict(path='scratch/helper.txt',content='second',overwrite=True)},edit]
        self.solve(actions)
        self.assertEqual((self.root/'scratch/helper.txt').read_text(),'third')
