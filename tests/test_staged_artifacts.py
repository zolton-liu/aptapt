"""Writer adaptation must preserve audit gates and actual candidate bytes."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qfa_agent.context import focused_preview
from qfa_agent.stages import run_stages
from qfa_agent.staged_artifacts import stage_writer, publish_artifacts
from qfa_agent.verification import AuditReport, structure, computation, reference_crosscheck


class StagedArtifactsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.output=self.root/'output'; self.output.mkdir()
        self.script=self.root/'solve.py'; self.script.write_text('# synthetic participant\n')
        self.calls=0
        self.bundle=None
        self.env=patch.dict(os.environ,{'OUTPUT_DIR':str(self.output),'QFA_VERIFICATION_REQUIRED':'1'})
        self.env.start(); self.addCleanup(self.env.stop)

    def writer(self, directory):
        self.calls+=1
        (directory/'results.json').write_text('{"total": 6}')
        (directory/'table.csv').write_text('x\n6\n')
        return {'total':6}

    def compute(self, inputs):
        self.bundle=stage_writer(self.writer)
        self.assertEqual(list(self.output.iterdir()),[])
        return self.bundle

    def audit(self, inputs, result):
        self.assertEqual(list(self.output.iterdir()),[])
        values=result['artifacts'].read_json('results.json')
        self.assertEqual(result['artifacts'].read_csv('table.csv').x.tolist(),[6])
        return AuditReport([structure(values,{'total':int}),computation(arrays=values),
                            reference_crosscheck(values['total'],lambda data:sum(data['x']),inputs,
                                                 description='sum original samples independently',atol=0)])

    def run_pipeline(self, *, compute=None, audit=None, write=None, run='fixture'):
        namespace={'load_inputs':lambda:{'x':[1,2,3]},'compute':compute or self.compute,
                   'audit':audit or self.audit,
                   'write_outputs':write or (lambda result:publish_artifacts(result['artifacts'])),
                   'OUTPUT_SCHEMA':{'results.json':{'format':'json','fields':{'total':int}},
                                    'table.csv':{'format':'csv','columns':{'x':'number'},'rows':1}}}
        evidence=self.root/(run+'.json')
        run_stages(namespace,self.script,evidence,run)
        return json.loads(evidence.read_text())

    def test_actual_audit_precedes_exact_byte_publication_no_rerun(self):
        evidence=self.run_pipeline()
        self.assertTrue(evidence['complete'])
        self.assertEqual(self.calls,1)
        self.assertEqual((self.output/'results.json').read_bytes(),b'{"total": 6}')
        self.assertEqual(evidence['stages'][2]['verification']['layers'],
                         {'structure':True,'computation':True,'cross_check':True})

    def test_failed_crosscheck_does_not_publish(self):
        def audit(inputs,result):
            return self.audit({'x':[10]},result)
        with self.assertRaisesRegex(ValueError,'verification failed'):
            self.run_pipeline(audit=audit)
        self.assertEqual(self.calls,1)
        self.assertEqual(list(self.output.iterdir()),[])

    def test_legacy_bool_audit_cannot_publish_in_strict_mode(self):
        with self.assertRaisesRegex(ValueError,'requires AuditReport'):
            self.run_pipeline(audit=lambda inputs,result:{'looks_good':True})
        self.assertEqual(list(self.output.iterdir()),[])

    def test_out_of_stage_calls_rejected(self):
        with self.assertRaisesRegex(ValueError,'compute stage'):
            stage_writer(self.writer)
        with self.assertRaisesRegex(ValueError,'write_outputs stage'):
            publish_artifacts(None)

    def test_early_publish_during_compute_rejected(self):
        def compute(inputs):
            result=stage_writer(self.writer)
            publish_artifacts(result['artifacts'])
            return result
        with self.assertRaisesRegex(ValueError,'write_outputs stage'):
            self.run_pipeline(compute=compute)
        self.assertEqual(list(self.output.iterdir()),[])

    def test_cross_run_bundle_reuse_rejected_even_when_checks_pass(self):
        # First run fails audit; preserve its actual candidate only in memory.
        with self.assertRaisesRegex(ValueError,'requires AuditReport'):
            self.run_pipeline(audit=lambda inputs,result:True)
        old=self.bundle
        with self.assertRaisesRegex(ValueError,'this staged run'):
            self.run_pipeline(compute=lambda inputs:old,run='second')
        self.assertEqual(list(self.output.iterdir()),[])

    def test_reject_reserved_files_symlinks_and_no_files(self):
        for kind in ('reserved','symlink','empty','none'):
            with self.subTest(kind=kind):
                def writer(path):
                    if kind=='reserved': (path/'reward.json').write_text('{}')
                    if kind=='symlink': (path/'x').symlink_to(self.script)
                    if kind=='empty': (path/'x').touch()
                    return {}
                with self.assertRaises(ValueError):
                    self.run_pipeline(compute=lambda inputs:stage_writer(writer),run=kind)
                self.assertEqual(list(self.output.iterdir()),[])

    def test_captured_bytes_do_not_follow_mutated_return_value(self):
        def audit(inputs,result):
            result['values']['total']=900
            return self.audit(inputs,result)
        self.run_pipeline(audit=audit)
        self.assertEqual(json.loads((self.output/'results.json').read_text()),{'total':6})

    def test_target_symlink_rejected_before_any_publication(self):
        def audit(inputs,result):
            report=self.audit(inputs,result)
            (self.output/'table.csv').symlink_to(self.script)
            return report
        with self.assertRaisesRegex(ValueError,'symlinks'):
            self.run_pipeline(audit=audit)
        self.assertFalse((self.output/'results.json').exists())
        self.assertEqual(self.script.read_text(),'# synthetic participant\n')

    def test_bound_lambda_location_not_shadowed_earlier_definition(self):
        source='def compute(inputs):\n    return {"x": 1}\n\ncompute = lambda inputs: {}\n'
        self.script.write_text(source)
        namespace={}; exec(compile(source,str(self.script),'exec'),namespace)
        with self.assertRaisesRegex(ValueError,'non-empty mapping'):
            self.run_pipeline(compute=namespace['compute'])
        evidence=json.loads((self.root/'fixture.json').read_text())
        preview=focused_preview('stage_evidence',evidence)
        location=preview['stages'][-1]['callable_location']
        self.assertTrue(location['location_verified'])
        self.assertEqual(location['line'],4)
        self.assertIn('lambda',location['source_context'])
        self.assertEqual(focused_preview('stage_evidence',preview)['stages'][-1]['callable_location'],location)
