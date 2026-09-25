from __future__ import annotations

import unittest

from qfa_agent.strategies import STRATEGIES
from qfa_agent.types import Action, ToolOutcome
from qfa_agent.workflow import WorkflowController, classify_failure


class WorkflowControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WorkflowController(STRATEGIES["risk-management"])

    def test_finish_is_blocked_until_explicit_validation(self) -> None:
        blocked = self.workflow.guard(Action("finish", {}))
        self.assertIsNotNone(blocked)
        self.assertIn("risk gate", blocked.summary)

        validated = ToolOutcome(True, "generic output checks passed")
        self.workflow.observe(Action("validate_outputs", {}), validated)
        self.assertIsNone(self.workflow.guard(Action("finish", {})))
        self.assertEqual(self.workflow.state.phase, "finish")

    def test_mutation_invalidates_an_earlier_validation(self) -> None:
        self.workflow.observe(
            Action("validate_outputs", {}), ToolOutcome(True, "generic output checks passed")
        )
        self.workflow.observe(
            Action("write_file", {"path": "scratch/solve.py"}),
            ToolOutcome(True, "wrote scratch/solve.py", mutated=True),
        )
        self.assertFalse(self.workflow.state.validation_passed)
        self.assertEqual(self.workflow.state.phase, "execute")

    def test_failed_run_is_classified_and_repetition_escalates(self) -> None:
        action = Action("run_python", {"script": "scratch/solve.py"})
        for _ in range(2):
            outcome = ToolOutcome(
                False,
                "Python program failed",
                {"output": "Traceback\nKeyError: 'price'"},
            )
            self.workflow.observe(action, outcome)
        self.assertEqual(self.workflow.state.last_failure_kind, "schema")
        self.assertEqual(self.workflow.state.repeated_failure_count, 2)
        self.assertIn("same failure recurred", self.workflow.guidance())

    def test_successful_run_enters_domain_audit(self) -> None:
        outcome = ToolOutcome(True, "Python program completed")
        self.workflow.observe(Action("run_python", {}), outcome)
        self.assertEqual(self.workflow.state.phase, "audit")
        self.assertIn("ES is at least VaR", outcome.data["workflow_guidance"])

    def test_controller_policy_rejection_is_not_recorded_as_solver_failure(self) -> None:
        outcome = ToolOutcome(
            False,
            "scratch/solve.py does not exist",
            {
                "controller_policy": "missing-solver",
                "policy_phase": "build",
                "required_next_step": "create the solver",
            },
        )
        self.workflow.observe(
            Action("run_python", {"script": "scratch/solve.py"}), outcome
        )
        self.assertEqual(self.workflow.state.phase, "build")
        self.assertEqual(self.workflow.state.failed_runs, 0)
        self.assertEqual(self.workflow.state.repair_attempts, 0)
        self.assertEqual(self.workflow.state.last_failure_kind, "")
        self.assertEqual(outcome.data["workflow"]["phase"], "build")

    def test_failure_taxonomy(self) -> None:
        self.assertEqual(
            classify_failure(ToolOutcome(False, "Python program failed", {"output": "SyntaxError"})),
            "syntax",
        )
        self.assertEqual(
            classify_failure(ToolOutcome(False, "Python program failed", {"timed_out": True})),
            "timeout",
        )
        self.assertEqual(
            classify_failure(ToolOutcome(False, "Python program failed", {"output": "NameError: x is not defined"})),
            "symbol",
        )


if __name__ == "__main__":
    unittest.main()
