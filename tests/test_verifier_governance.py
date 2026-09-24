"""Fail-closed receipt/schema tests; no public task answers."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from qfa_agent.experience import ExperienceStore
from qfa_agent.verification import artifact_structure, pricing_crosscheck


class VerifierGovernanceTests(unittest.TestCase):
    def test_pricing_dimension_and_tolerance_errors_are_not_method_failures(self):
        args = dict(grids=(10,20,40),boundary_actual=[0],boundary_reference=[0],
                    reference_description='synthetic zero boundary',atol=.02)
        shape = pricing_crosscheck([1], [1,2], [1], **args)
        self.assertFalse(shape.passed)
        self.assertEqual(shape.feedback,'implementation')
        bad_tolerance = pricing_crosscheck([1], [1], [1], **{**args,'atol':-1})
        self.assertFalse(bad_tolerance.passed)
        self.assertEqual(bad_tolerance.feedback,'implementation')
        convergence = pricing_crosscheck([1], [2], [8], **args)
        self.assertFalse(convergence.passed)
        self.assertEqual(convergence.feedback,'method_review')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = ExperienceStore(self.root/'experience')

    def test_malformed_catalog_is_not_a_solver_crash_or_retrievable_fact(self):
        self.store.root.mkdir()
        for payload in [None, [], 1, {'validation':None}, {'validation':{'cases':None}},
                        {'status':'verified','validation':{'cases':[None]}}]:
            with self.subTest(payload=payload):
                self.store.path.write_text(json.dumps(payload))
                self.assertEqual(self.store.retrieve(category='factor-research',failure_kind='alignment'), [])

    def test_promotion_rejects_disabled_assertions(self):
        self.store.candidate('Synthetic shape mismatch, unconfirmed real cause')
        with patch('qfa_agent.experience.sys.flags', SimpleNamespace(optimize=1)):
            with self.assertRaisesRegex(ValueError,'assertions enabled'):
                self.store.verify()
        self.assertEqual(json.loads(self.store.path.read_text())['status'],'candidate')

    def test_promotion_engine_change_requires_revalidation(self):
        self.store.candidate('Synthetic shape mismatch'); self.store.verify()
        self.assertEqual(len(self.store.retrieve(category='factor-research',failure_kind='alignment')), 1)
        with patch('qfa_agent.experience._engine_binding',return_value='new-engine'):
            self.assertEqual(self.store.retrieve(category='factor-research',failure_kind='alignment'), [])

    def test_json_overflow_and_nested_nonfinite_are_not_valid_structure(self):
        spec = {'result.json':{'format':'json','fields':{'results':list}}}
        for value in ['{"results":[1e309]}', '{"results":[{"x":NaN}]}', '{"results":[-1e309]}']:
            with self.subTest(value=value):
                (self.root/'result.json').write_text(value)
                self.assertFalse(artifact_structure(self.root, spec).passed)
        (self.root/'result.json').write_text('{"results":[{"x":1.25}]}')
        self.assertTrue(artifact_structure(self.root,spec).passed)

    def test_csv_schema_checks_actual_cells_and_row_count(self):
        spec = {'result.csv':{'format':'csv','columns':{'date':'string','value':'number'},'rows':1}}
        for text in ['date,value\n', 'date,value\n2020-01-01,NaN\n', 'date,value\n2020-01-01,1,2\n',
                     'value,date\n1,2020-01-01\n']:
            (self.root/'result.csv').write_text(text)
            self.assertFalse(artifact_structure(self.root,spec).passed)
        (self.root/'result.csv').write_text('date,value\n2020-01-01,1.25\n')
        self.assertTrue(artifact_structure(self.root,spec).passed)
