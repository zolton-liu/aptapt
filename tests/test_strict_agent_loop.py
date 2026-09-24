"""Whole-controller verification flows using scripted replies, not real model calls."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qfa_agent.agent import AgentConfig, CodingAgent
from qfa_agent.model import ModelReply
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.workspace import TaskWorkspace


SOURCE = '''QFA_STAGED = True
QFA_VERIFICATION = "three-layer-v1"
OUTPUT_SCHEMA = {"results.json":{"format":"json","fields":{"var":float,"es":float}}}
from qfa_agent.contracts import input_path
from qfa_agent.verification import AuditReport,structure,computation,risk_crosscheck
def load_inputs():
    return json.loads(input_path("losses.json").read_text())
def compute(inputs):
    losses = np.asarray(inputs["losses"],dtype=float)
    var = float(np.quantile(losses,inputs["alpha"]))
    es = float(losses[losses >= var].mean())
    return {"var":var,"es":es + ERROR_OFFSET}
def audit(inputs,result):
    return AuditReport([structure(result,{"var":float,"es":float}),
        computation(arrays=result),risk_crosscheck(inputs["losses"],result["var"],result["es"],
        alpha=inputs["alpha"],quantile_rule="linear",es_rule="mean_at_or_above_var")])
def write_outputs(result):
    (output_dir/"results.json").write_text(json.dumps(result,allow_nan=False))
'''


class StrictAgentLoopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ('input','output','scratch'): (self.root/folder).mkdir()
        (self.root/'input/card.toml').write_text('[task]\nid="synthetic-strict-loop"\n[agent]\ntimeout_sec=120\n')
        (self.root/'input/instruction.md').write_text('Calculate empirical loss statistics and write output/results.json.')
        (self.root/'input/losses.json').write_text('{"losses":[0,1,2,3],"alpha":0.5}')
        self.workspace = TaskWorkspace(*(self.root/p for p in ('input','output','scratch')))

    def write(self, offset, overwrite=False):
        return {'tool':'write_file','arguments':{'path':'scratch/solve.py',
            'content':SOURCE.replace('ERROR_OFFSET',str(offset)),'overwrite':overwrite}}

    def solve(self, actions):
        class ScriptedModel:
            name='strict-loop-fixture'
            def __init__(self): self.actions=iter(actions); self.requests=[]
            def complete(self, messages, *, timeout_sec):
                self.requests.append(messages)
                return ModelReply(next(self.actions))
        model=ScriptedModel()
        with patch.dict(os.environ,{'QFA_VERIFICATION_REQUIRED':'1','QFA_STAGE_PIPELINE':'1'}):
            result=CodingAgent(model,AgentConfig(starters_enabled=False,max_steps=len(actions),
                observation_max_chars=1500,reserve_sec=2)).solve(load_task(self.root/'input'),
                    self.workspace,Trajectory(self.root/'trajectory.jsonl'))
        return result,model

    def test_actual_audit_and_artifact_checks_allow_finish_after_auto_run(self):
        result,model=self.solve([self.write(0),{'tool':'finish','arguments':{}}])
        self.assertTrue(result.succeeded,result.message)
        self.assertEqual(len(model.requests),2)
        self.assertFalse(result.starter_used)
        self.assertEqual(json.loads((self.root/'output/results.json').read_text()),{'var':1.5,'es':2.5})
        records=[json.loads(p.read_text()) for p in (self.root/'scratch/.agent/stages').glob('*.json')]
        self.assertEqual(len(records),1)
        self.assertTrue(records[0]['complete'])
        self.assertEqual(records[0]['stages'][2]['verification']['layers'],
                         {'structure':True,'computation':True,'cross_check':True})
        self.assertTrue(records[0]['stages'][3]['artifact_check']['passed'])

    def test_crosscheck_failure_cannot_finish_or_escape_through_budget_end(self):
        result,_=self.solve([self.write(1),{'tool':'finish','arguments':{}}])
        self.assertFalse(result.succeeded)
        self.assertFalse((self.root/'output/results.json').exists())
        self.assertIn('verification gate',result.message)

    def test_repair_keeps_failed_evidence_and_revalidates_new_source(self):
        result,model=self.solve([self.write(1),self.write(0,overwrite=True),{'tool':'finish','arguments':{}}])
        self.assertTrue(result.succeeded,result.message)
        self.assertEqual(len(model.requests),3)
        records=[json.loads(p.read_text()) for p in (self.root/'scratch/.agent/stages').glob('*.json')]
        self.assertEqual(len(records),2)
        self.assertEqual(len({r['source_sha256'] for r in records}),2)
        self.assertEqual(sorted(r['complete'] for r in records),[False,True])
        memory=json.loads((self.root/'scratch/.agent/memory.json').read_text())
        self.assertTrue(memory['workflow']['solver_stages']['complete'])
        self.assertTrue(any(f['status']=='resolved' for f in memory['all_bounded_failures']))

    def test_writer_adapter_runs_through_actual_runtime_and_controller(self):
        source=SOURCE.replace('ERROR_OFFSET','0').replace('def compute(inputs):','def calculate(inputs):')
        source=source.replace('def audit(inputs,result):', '''from qfa_agent.staged_artifacts import stage_writer,publish_artifacts
def legacy_writer(inputs,directory):
    values=calculate(inputs)
    (directory/"results.json").write_text(json.dumps(values,allow_nan=False))
    return values
def compute(inputs):
    return stage_writer(lambda directory:legacy_writer(inputs,directory))
def audit(inputs,result):
    result=result["artifacts"].read_json("results.json")''')
        source=source.replace('(output_dir/"results.json").write_text(json.dumps(result,allow_nan=False))',
                              'publish_artifacts(result["artifacts"])')
        result,model=self.solve([{'tool':'write_file','arguments':{'path':'scratch/solve.py','content':source}},
                                 {'tool':'finish','arguments':{}}])
        self.assertTrue(result.succeeded,result.message)
        self.assertFalse(result.starter_used)
        self.assertEqual(len(model.requests),2)
        self.assertEqual(json.loads((self.root/'output/results.json').read_text()),{'var':1.5,'es':2.5})
