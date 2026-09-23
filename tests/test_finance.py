from __future__ import annotations

import unittest

from qfa_agent.finance import finance_hints


class FinanceHintTests(unittest.TestCase):
    def test_bootstrap_hint_calls_for_par_repricing(self) -> None:
        hints = finance_hints("Bootstrap a zero-coupon curve from par coupon rates")
        combined = " ".join(hints).lower()
        self.assertIn("sum the present values", combined)
        self.assertIn("reprice every input par bond", combined)
        self.assertIn("z(t-1)", combined)
        self.assertIn("maturity keys numeric", combined)
        self.assertIn("dict.get(time, 0)", combined)
        self.assertIn("candidate z(t)", combined)
        self.assertIn("root finder or bisection", combined)
        self.assertIn("scipy.optimize.brentq", combined)
        self.assertIn("residual(candidate_z)", combined)
        self.assertIn("known_z: dict[float, float]", combined)


if __name__ == "__main__":
    unittest.main()
