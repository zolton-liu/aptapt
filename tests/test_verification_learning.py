from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from qfa_agent.verification import (AuditReport, Check, VerificationFailure, structure,
    computation, regression_crosscheck, risk_crosscheck, pricing_crosscheck,
    ledger_crosscheck, artifact_structure, reference_crosscheck)
from qfa_agent.experience import ExperienceStore
from qfa_agent.experience_rules import RULE, validate_rule
from qfa_agent.memory import WorkingMemory
from qfa_agent.strategies import STRATEGIES
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import WorkflowController
from qfa_agent.workspace import TaskWorkspace, ToolRouter


class VerificationLearningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for p in ('input', 'output', 'scratch'): (self.root / p).mkdir()
        self.w = TaskWorkspace(*(self.root / p for p in ('input', 'output', 'scratch')))

    def base_checks(self):
        return [structure({'value': 1.0}, {'value': float}), computation(arrays={'value': [1.0]})]

    def pricing(self, *, bad=False):
        return pricing_crosscheck([1.1], [1.025], [4 if bad else 1.00625], grids=(10,20,40),
                                  boundary_actual=[0.0], boundary_reference=[0.0],
                                  reference_description='fixture zero-payoff boundary', atol=0.02)

    def test_three_layers_required_and_missing_is_not_pass(self):
        with self.assertRaises(ValueError): AuditReport(self.base_checks())
        with self.assertRaises(ValueError): AuditReport([True, True, True])
        with self.assertRaises(ValueError): Check('cross_check', 'fake', True, 'I believe it')
        report = AuditReport([*self.base_checks(), self.pricing()])
        self.assertTrue(report.as_dict()['passed'])
        self.assertEqual(set(report.as_dict()['layers']), {'structure','computation','cross_check'})
        json.dumps(report.as_dict(), allow_nan=False)

    def test_structure_rejects_wrong_type_shape_or_missing_field(self):
        for values, fields, shapes in [({'x':True},{'x':int},None), ({},{'x':float},None),
                                      ({'x':[1,2]},{'x':list},{'x':(3,)})]:
            self.assertFalse(structure(values, fields, shapes=shapes).passed)

    def test_computation_rejects_nonfinite_alignment_and_invalid_matrix(self):
        self.assertFalse(computation(arrays={'x':[float('nan')]}).passed)
        a = pd.Series([1,2], index=['a','b']); b = pd.Series([1,2], index=['b','a'])
        self.assertFalse(computation(arrays={}, aligned=(a,b)).passed)
        self.assertFalse(computation(arrays={}, correlations=([[1,2],[2,1]],)).passed)

    def regression(self, bad=False, shuffled=False):
        idx = pd.date_range('2020-01-01', periods=3)
        X = pd.DataFrame({'const':1.,'x':[1.,2.,3.]},index=idx)
        beta = pd.Series({'const':1.,'x':2.})
        y = pd.Series([3.,5.,8.],index=idx)
        prediction = pd.Series([3.,5.,7.],index=idx)
        residual = pd.Series([0.,0.,1.],index=idx)
        if bad: prediction.iloc[1] = 6
        if shuffled: y = y.iloc[::-1]
        return regression_crosscheck(X,y,beta,prediction,residual)

    def test_regression_recomputes_not_just_shapes(self):
        self.assertTrue(self.regression().passed)
        self.assertFalse(self.regression(bad=True).passed)
        self.assertFalse(self.regression(shuffled=True).passed)

    def test_risk_recomputes_declared_quantile_and_tail_conventions(self):
        self.assertTrue(risk_crosscheck([0,1,2,3],1.5,2.5,alpha=.5,quantile_rule='linear',es_rule='fractional_tail').passed)
        self.assertFalse(risk_crosscheck([0,1,2,3],1.5,3.,alpha=.5,quantile_rule='linear',es_rule='fractional_tail').passed)
        self.assertTrue(risk_crosscheck([0,1,1,4],1,2,alpha=.5,quantile_rule='inverted_cdf',es_rule='mean_at_or_above_var').passed)
        self.assertFalse(risk_crosscheck([0,1],0,1,alpha=.5,quantile_rule='guessed',es_rule='fractional_tail').passed)

    def test_pricing_failure_requests_method_review_not_proven_root_cause(self):
        report = AuditReport([*self.base_checks(), self.pricing(bad=True)])
        self.assertEqual(report.as_dict()['feedback_route'],'method_review')
        with self.assertRaises(VerificationFailure): report.require_pass()
        self.assertIn('implementation',report.checks[-1].scope)

    def test_ledger_replay_catches_internally_consistent_but_wrong_cash(self):
        dates = pd.date_range('2020-01-01',periods=3)
        marks = pd.DataFrame({'A':[10.,11.,12.]},index=dates)
        trades = [{'time':dates[0], 'asset':'A','quantity':10.,'price':10.,'fee':1.},
                  {'time':dates[2], 'asset':'A','quantity':-5.,'price':12.,'fee':1.}]
        cash = pd.Series([99.,99.,158.],index=dates)
        equity = pd.Series([199.,209.,218.],index=dates)
        self.assertTrue(ledger_crosscheck(trades,marks,cash,equity,initial_cash=200.,initial_positions={}).passed)
        self.assertFalse(ledger_crosscheck(trades,marks,cash+1,equity+1,initial_cash=200.,initial_positions={}).passed)
        trades[0]['dividend'] = 2
        self.assertFalse(ledger_crosscheck(trades,marks,cash,equity,initial_cash=200.,initial_positions={}).passed)

    def test_files_checked_after_write_not_only_result_dict(self):
        out = self.root/'output'
        spec = {'result.json':{'format':'json','fields':{'value':float}}}
        self.assertFalse(artifact_structure(out,spec).passed)
        (out/'result.json').write_text('{"value": "1"}')
        self.assertFalse(artifact_structure(out,spec).passed)
        (out/'result.json').write_text('{"value": 1.0}')
        self.assertTrue(artifact_structure(out,spec).passed)

    def test_custom_reference_runs_and_discloses_weaker_provenance(self):
        called=[]
        def reference(values): called.append(True); return [sum(values)]
        check=reference_crosscheck([4],reference,[1,2],description='separate scalar sum',atol=0)
        self.assertFalse(check.passed); self.assertEqual(called,[True])
        self.assertIn('not established',check.scope)

    def test_workflow_routes_structured_method_evidence_and_records_revision(self):
        workflow=WorkflowController(STRATEGIES['derivatives-pricing'])
        report=AuditReport([*self.base_checks(),self.pricing(bad=True)]).as_dict()
        outcome=ToolOutcome(False,'Python program failed',{'returncode':1,'stage_evidence':{
            'mode':'staged','source_sha256':'original','stages':[{'name':'audit','status':'failed','verification':report}]}})
        workflow.observe(Action('run_python',{'script':'scratch/solve.py'}),outcome)
        self.assertEqual(workflow.feedback_route,'method_review')
        self.assertIn('Method review',workflow.guidance())
        action=Action('revise_method',{'hypothesis':'Grid too coarse, provisional', 'evidence':'refinement discrepancy',
                                       'change':'Increase allowed grid resolution','falsification':'Same unchanged tolerance'})
        result=ToolRouter(self.w).dispatch(action)
        workflow.observe(action,result)
        self.assertTrue(result.ok)
        self.assertEqual(workflow.snapshot()['method_revisions'][0]['status'],'hypothesis_not_validated')
        self.assertIsNotNone(workflow.guard(Action('finish')))

    def test_plain_runtime_error_not_blindly_called_method_failure(self):
        workflow=WorkflowController(STRATEGIES['factor-research'])
        workflow.observe(Action('run_python'),ToolOutcome(False,'ValueError: operands could not be broadcast together',{'returncode':1}))
        self.assertEqual(workflow.feedback_route,'implementation')
        workflow.observe(Action('run_python'),ToolOutcome(False,'FloatingPointError: NaN',{'returncode':1}))
        self.assertEqual(workflow.feedback_route,'undetermined')

    def test_candidate_not_retrieved_and_full_variants_needed(self):
        store=ExperienceStore(self.root/'experience')
        store.candidate('Observed y=2516, X=2515; root cause unknown')
        self.assertEqual(store.retrieve(category='factor-research',failure_kind='alignment'),[])
        record=store.verify()
        self.assertEqual(record['status'],'verified')
        self.assertEqual(len(record['validation']['cases']),5)
        self.assertFalse(record['actual_task_cause_verified'])
        self.assertEqual(len(store.retrieve(category='factor-research',failure_kind='alignment')),1)
        self.assertEqual(store.retrieve(category='derivatives-pricing',failure_kind='path'),[])
        record['validation']['cases'].pop()
        store.path.write_text(json.dumps(record))
        self.assertEqual(store.retrieve(category='factor-research',failure_kind='alignment'),[])

    def test_failed_variant_and_modified_semantics_block_promotion(self):
        store=ExperienceStore(self.root/'experience');store.candidate('shape failure')
        cases=validate_rule(); cases[-1]['passed']=False
        with patch('qfa_agent.experience_rules.validate_rule',return_value=cases):
            self.assertEqual(store.verify()['status'],'rejected')
        record=json.loads(store.path.read_text());record['rule']['repair']='truncate both arrays'
        store.path.write_text(json.dumps(record))
        with self.assertRaises(ValueError):store.verify()

    def test_catalog_source_change_invalidates_receipt(self):
        store=ExperienceStore(self.root/'experience');store.candidate('shape failure');store.verify()
        with patch('qfa_agent.experience._binding',return_value='changed'):
            self.assertEqual(store.retrieve(category='factor-research',failure_kind='alignment'),[])

    def test_memory_retrieves_frozen_catalog_only_for_matching_evidence(self):
        store=ExperienceStore(self.root/'experience');store.candidate('shape failure');store.verify()
        with patch.dict(os.environ,{'QFA_EXPERIENCE_DIR':str(store.root)}):
            memory=WorkingMemory(self.w,'task','factor-research')
        memory.observe(1,Action('run_python',{'script':'scratch/solve.py'}),
                       ToolOutcome(False,'ValueError: could not be broadcast together'), 'code')
        record=json.loads(store.path.read_text());record['status']='candidate';store.path.write_text(json.dumps(record))
        cp=memory.checkpoint(workflow={},required_files=(),solver_digest='code',budget={})
        self.assertEqual(len(cp['verified_cross_task_experience']),1) # frozen at task start
        self.assertIn('must be checked',cp['verified_cross_task_experience'][0]['applicability_to_current_task'])

    def test_real_runtime_structured_audit_blocks_write_and_preserves_report(self):
        source='''QFA_STAGED = True
QFA_VERIFICATION = 'three-layer-v1'
OUTPUT_SCHEMA = {'result.json':{'format':'json','fields':{'value':float}}}
from qfa_agent.verification import AuditReport,structure,computation,risk_crosscheck
def load_inputs(): return {'losses':[0,1,2,3]}
def compute(inputs): return {'value':1.5,'es':3.0}
def audit(inputs,result):
    return AuditReport([structure(result,{'value':float,'es':float}),
        computation(arrays=result), risk_crosscheck(inputs['losses'],result['value'],result['es'],
        alpha=.5,quantile_rule='linear',es_rule='fractional_tail')])
def write_outputs(result): (output_dir/'result.json').write_text(json.dumps(result))
'''
        self.w.write_file('scratch/solve.py',source)
        result=self.w.run_python('scratch/solve.py',[],20)
        self.assertFalse(result.ok);self.assertFalse((self.root/'output/result.json').exists())
        receipt=result.data['stage_evidence']['stages'][-1]['verification']
        self.assertEqual(receipt['layers'],{'structure':True,'computation':True,'cross_check':False})
        self.w.write_file('scratch/solve.py',source.replace("'es':3.0","'es':2.5"),overwrite=True)
        result=self.w.run_python('scratch/solve.py',[],20)
        self.assertTrue(result.ok,result.data['output'])
        self.assertTrue(result.data['stage_evidence']['stages'][-1]['artifact_check']['passed'])
        workflow = WorkflowController(STRATEGIES['risk-management'])
        workflow.observe(Action('run_python', {'script':'scratch/solve.py'}), result)
        workflow.state.validation_passed = True
        with patch.dict(os.environ, {'QFA_VERIFICATION_REQUIRED':'1'}):
            self.assertIsNone(workflow.guard(Action('finish')))
            workflow.observe(Action('write_file', {'path':'scratch/solve.py'}), ToolOutcome(True,'edited',mutated=True))
            self.assertIsNotNone(workflow.guard(Action('finish')))

    def test_strict_mode_rejects_legacy_bool_audit_even_without_opt_in_marker(self):
        self.w.write_file('scratch/solve.py', '''QFA_STAGED = True
def load_inputs(): return {'x':1}
def compute(inputs): return inputs
def audit(inputs,result): return {'fake':True}
def write_outputs(result): pass
''')
        with patch.dict(os.environ, {'QFA_VERIFICATION_REQUIRED':'1'}):
            result = self.w.run_python('scratch/solve.py', [], 20)
            self.assertFalse(result.ok)
            self.assertIn('AuditReport', result.data['output'])
            workflow = WorkflowController(STRATEGIES['risk-management'])
            workflow.state.validation_passed = True
            self.assertIsNotNone(workflow.guard(Action('finish')))

    def test_strict_mode_blocks_budget_end_bypass(self):
        from qfa_agent.agent import CodingAgent, AgentConfig
        from qfa_agent.model import ModelReply
        from qfa_agent.task import load_task
        from qfa_agent.trace import Trajectory
        class FakeModel:
            name='fixture'
            def complete(self, messages, *, timeout_sec):
                return ModelReply({'tool':'write_file','arguments':{'path':'output/results.json','content':'{"x":1}'}})
        (self.root/'input/instruction.md').write_text('Write output/results.json.')
        (self.root/'input/card.toml').write_text('[task]\nid="fixture"\n[agent]\ntimeout_sec=120\n')
        with patch.dict(os.environ, {'QFA_VERIFICATION_REQUIRED':'1'}):
            result = CodingAgent(FakeModel(), AgentConfig(starters_enabled=False,max_steps=1)).solve(
                load_task(self.root/'input'), self.w, Trajectory(self.root/'trace.jsonl'))
        self.assertFalse(result.succeeded)
        self.assertIn('verification gate', result.message)
