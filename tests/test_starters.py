from __future__ import annotations

import unittest
from types import SimpleNamespace

from qfa_agent.starters import starter_for, trusted_operator_program_for


class StarterTests(unittest.TestCase):
    def test_cliquet_trusted_operator_program_requires_exact_input(self) -> None:
        task = SimpleNamespace(
            safe_instruction=(
                "Price a cliquet as a strip of ATM forward-start options and write outputs."
            )
        )
        program = trusted_operator_program_for(
            task, ("environment/data/spy_daily.csv",)
        )
        self.assertIsNotNone(program)
        assert program is not None
        self.assertIn("write_cliquet_outputs", program.content)
        self.assertIn("rate=0.05", program.content)
        self.assertIsNone(
            trusted_operator_program_for(task, ("environment/data/other.csv",))
        )

    def test_selects_brinson_attribution_starter(self) -> None:
        task = SimpleNamespace(
            safe_instruction=(
                "Brinson-Fachler Sector Attribution with interaction and "
                "weight_snapshot_march outputs."
            )
        )
        starter = starter_for(
            task,
            (
                "environment/data/sector_etfs.csv",
                "environment/data/params.json",
                "environment/data/cash_rates.csv",
            ),
        )
        self.assertIsNotNone(starter)
        assert starter is not None
        compile(starter.content, starter.path, "exec")
        self.assertIn("ending_values", starter.content)
        self.assertIn("active_weight", starter.content)
        self.assertNotIn("0.20773056", starter.content)

    def test_selects_standard_var_starter(self) -> None:
        task = SimpleNamespace(
            safe_instruction=(
                "Standard VaR Methods Comparison with Age-weighted historical simulation "
                "and a Kupiec backtest."
            )
        )
        starter = starter_for(
            task,
            ("environment/data/pair_aapl_jpm_daily.csv", "instruction.md"),
        )
        self.assertIsNotNone(starter)
        assert starter is not None
        compile(starter.content, starter.path, "exec")
        self.assertIn("student_risk", starter.content)
        self.assertIn("kupiec_pvalue", starter.content)
        self.assertNotIn("30594.61", starter.content)

    def test_selects_general_ohlc_volatility_starter(self) -> None:
        task = SimpleNamespace(
            safe_instruction=(
                "OHLC Realized Volatility Estimators using Rogers-Satchell and Yang-Zhang."
            )
        )
        starter = starter_for(
            task,
            ("environment/data/spy_daily.csv", "instruction.md"),
        )
        self.assertIsNotNone(starter)
        assert starter is not None
        compile(starter.content, starter.path, "exec")
        self.assertIn("write_ohlc_volatility_outputs", starter.content)
        self.assertNotIn("687.06", starter.content)

    def test_selects_general_option_parity_audit_starter(self) -> None:
        task = SimpleNamespace(
            safe_instruction=(
                "Audit put-call parity using synthetic_forward_bid and report "
                "implied_borrow_rate."
            )
        )
        inventory = (
            "environment/data/option_quotes.csv",
            "environment/data/spot.json",
            "environment/data/carry_inputs.json",
            "environment/data/audit_rules.json",
        )
        starter = starter_for(task, inventory)
        self.assertIsNotNone(starter)
        assert starter is not None
        compile(starter.content, starter.path, "exec")
        self.assertIn("forward_too_high", starter.content)
        self.assertIn("quote_age_sec", starter.content)
        self.assertNotIn("100.25", starter.content)

    def test_selects_general_bootstrap_starter(self) -> None:
        task = SimpleNamespace(
            safe_instruction="Bootstrap a zero-coupon yield curve from par coupon rates."
        )
        starter = starter_for(
            task,
            ("card.toml", "environment/data/curve_data.json", "instruction.md"),
        )
        self.assertIsNotNone(starter)
        assert starter is not None
        compile(starter.content, starter.path, "exec")
        self.assertIn("brentq", starter.content)
        self.assertIn("par repricing self-check", starter.content)
        self.assertNotIn("0.045", starter.content)

    def test_skips_unrelated_tasks(self) -> None:
        task = SimpleNamespace(safe_instruction="Compute arithmetic returns.")
        self.assertIsNone(
            starter_for(task, ("environment/data/curve_data.json",))
        )


if __name__ == "__main__":
    unittest.main()
