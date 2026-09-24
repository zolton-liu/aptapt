import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qfa_agent.agent import AgentConfig, CodingAgent
from qfa_agent.prompts import system_prompt
from qfa_agent.strategies import RoutingDecision, STRATEGIES
from qfa_agent.task import load_task
from qfa_agent.trace import Trajectory
from qfa_agent.verification_prompt import DOMAIN_APIS, verification_instructions
from qfa_agent.workspace import TaskWorkspace


class VerificationPromptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('input','output','scratch'): (self.root/name).mkdir()
        (self.root/'input/card.toml').write_text('[task]\nid="fixture"\n[agent]\ntimeout_sec=120\n')
        (self.root/'input/instruction.md').write_text('Write output/results.json.')

    def prompt(self, category, inventory=()):
        return system_prompt(load_task(self.root/'input'), inventory, 'fixture input profile',
                             RoutingDecision(STRATEGIES[category],category,'fixture',1.0))

    def test_strict_contract_is_present_for_all_routes(self):
        with patch.dict(os.environ,{'QFA_VERIFICATION_REQUIRED':'1'}):
            for category in STRATEGIES:
                with self.subTest(category=category):
                    prompt = self.prompt(category)
                    self.assertIn('audit returns AuditReport', prompt)
                    self.assertIn('OUTPUT_SCHEMA', prompt)
                    self.assertIn('not the official checker', prompt)

    def test_strict_template_preserves_implementation_without_audit_exemption(self):
        (self.root/'input/instruction.md').write_text('Fix old API migration; write output/results.json.')
        with patch.dict(os.environ,{'QFA_VERIFICATION_REQUIRED':'1'}):
            prompt = self.prompt('software-repair', ('environment/template.py',))
        self.assertIn('copy_file',prompt)
        self.assertIn('audit returns AuditReport',prompt)
        self.assertIn('legacy run cannot satisfy',prompt)

    def test_operator_adapter_also_gets_strict_verifier_contract(self):
        (self.root/'input/instruction.md').write_text('Compute Black-Scholes Greeks; write output/results.json.')
        with patch.dict(os.environ,{'QFA_VERIFICATION_REQUIRED':'1'}):
            prompt = self.prompt('derivatives-pricing')
        self.assertIn('black_scholes_metrics',prompt)
        self.assertIn('audit returns AuditReport',prompt)

    def test_only_relevant_domain_api_details_are_injected(self):
        for category, explanation in DOMAIN_APIS.items():
            text = verification_instructions(category)
            self.assertIn(explanation,text)
            for other, detail in DOMAIN_APIS.items():
                if other != category: self.assertNotIn(detail,text)
            self.assertIn('reference_crosscheck',text)
            self.assertIn('All four domain helpers',text)

    def test_conflicting_strict_configuration_fails_before_model_call(self):
        class NoModel:
            name = 'must-not-run'
            def complete(self, *args, **kwargs):
                raise AssertionError('model called despite invalid configuration')
        workspace = TaskWorkspace(*(self.root/name for name in ('input','output','scratch')))
        with patch.dict(os.environ, {'QFA_VERIFICATION_REQUIRED':'1','QFA_STAGE_PIPELINE':'0'}):
            with self.assertRaisesRegex(ValueError,'requires QFA_STAGE_PIPELINE'):
                CodingAgent(NoModel(),AgentConfig(starters_enabled=False)).solve(
                    load_task(self.root/'input'),workspace,Trajectory(self.root/'trace.jsonl'))
