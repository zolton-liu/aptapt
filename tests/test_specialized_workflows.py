from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qfa_agent.american_fd import write_american_option_fd_outputs
from qfa_agent.calibration_workflows import (
    write_fama_french_outputs,
    write_mean_reverting_jump_diffusion_outputs,
)
from qfa_agent.derivatives_workflows import (
    _black_scholes_price,
    write_implied_volatility_approximation_outputs,
    write_variance_swap_outputs,
)


class SpecializedWorkflowTests(unittest.TestCase):
    def test_implied_volatility_bundle_reprices_all_grid_points(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = write_implied_volatility_approximation_outputs(Path(temporary))
            summary = bundle["summary"]
            self.assertEqual(summary["n_points"], 75)
            self.assertTrue(summary["all_newton_converged"])
            self.assertLess(summary["rmse_newton"], 1.0e-8)
            self.assertAlmostEqual(summary["rmse_bs_atm"], 0.03418723590366654)
            self.assertAlmostEqual(summary["rmse_li_nonatm"], 0.005730051959888682)
            self.assertAlmostEqual(summary["rmse_cmh"], 0.28387655126244243)

    def test_variance_swap_bundle_reconciles_counts_and_pnl(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            params = {
                "spot": 100.0, "risk_free_rate": 0.0, "dividend_yield": 0.0,
                "target_expiry": "90d", "expiries": {"90d": 0.5},
                "forward_prices": {"90d": 100.0},
                "filters": {"min_volume": 10, "max_relative_spread": 0.5,
                            "exclude_zero_bid": True},
            }
            (root / "params.json").write_text(json.dumps(params), encoding="utf-8")
            rows = []
            for strike in range(80, 121, 5):
                call = _black_scholes_price(100, strike, 0.5, 0, 0, 0.2, "call")
                put = _black_scholes_price(100, strike, 0.5, 0, 0, 0.2, "put")
                rows.append({
                    "expiry": "90d", "strike": strike,
                    "call_bid": call * 0.99, "call_ask": call * 1.01,
                    "put_bid": put * 0.99, "put_ask": put * 1.01,
                    "volume": 100, "open_interest": 1000,
                })
            pd.DataFrame(rows).to_csv(root / "chain.csv", index=False)
            bundle = write_variance_swap_outputs(
                root / "chain.csv", root / "params.json", root / "output"
            )
            result = bundle["results"]
            self.assertEqual(result["n_strikes_kept"], len(rows))
            self.assertEqual(
                result["n_otm_puts_kept"] + result["n_otm_calls_kept"], len(rows)
            )
            for scenario in result["scenarios"]:
                self.assertAlmostEqual(
                    scenario["pnl"],
                    10_000 * (scenario["realized_variance"] - result["fair_variance_strike"]),
                )

    def test_jump_diffusion_bundle_has_consistent_moments(self) -> None:
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(7)
            values = [-3.5]
            for _ in range(499):
                values.append(-3.5 + 0.98 * (values[-1] + 3.5) + rng.normal(0, 0.01))
            pd.DataFrame({"DGS10": np.exp(values) * 100}).to_csv(root / "rates.csv", index=False)
            bundle = write_mean_reverting_jump_diffusion_outputs(
                root / "rates.csv", root / "output", horizons=(0.05, 0.10),
                paths=3000, seed=11,
            )
            calibration = bundle["calibration"]
            self.assertEqual(calibration["n_changes"], calibration["n_obs"] - 1)
            self.assertGreater(calibration["kappa"], 0)
            self.assertTrue((bundle["conditional_moments"]["Var_X"] > 0).all())

    def test_fama_french_bundle_aligns_and_reconstructs(self) -> None:
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(19)
            dates = pd.date_range("2020-01-01", periods=281, freq="B")
            factor_values = rng.normal(0, [0.01, 0.006, 0.006], size=(280, 3))
            factors = pd.DataFrame(
                factor_values, index=dates[1:], columns=["Mkt-RF", "SMB", "HML"]
            )
            factors["RF"] = 0.00005
            tickers = ["AAPL", "AMZN", "BRK-B", "GOOGL", "GS", "JPM", "MCD", "MSFT", "UNH", "V"]
            prices = pd.DataFrame(index=dates)
            for index, ticker in enumerate(tickers):
                beta = np.array([0.8 + 0.04 * index, -0.2 + 0.03 * index, 0.1 - 0.02 * index])
                returns = 0.00005 + 0.00001 * index + factor_values @ beta + rng.normal(0, 0.004, 280)
                prices[ticker] = np.r_[100.0, 100.0 * np.cumprod(1.0 + returns)]
            prices.to_csv(root / "prices.csv")
            factors.to_csv(root / "factors.csv")
            bundle = write_fama_french_outputs(
                root / "prices.csv", root / "factors.csv", root / "output"
            )
            self.assertEqual(bundle["results"]["meta"]["trading_days"], 280)
            self.assertEqual(len(bundle["results"]["stocks"]), 10)
            self.assertEqual(len(bundle["rolling_betas"]), 29)
            self.assertEqual(len(list((root / "output").iterdir())), 8)

    def test_american_fd_small_grid_preserves_finance_identities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = write_american_option_fd_outputs(
                temporary, fine_stock_steps=60, fine_time_steps=120,
                coarse_stock_steps=30, coarse_time_steps=60,
            )
            summary = bundle["summary"]
            self.assertTrue(summary["american_geq_european_put"])
            self.assertTrue(summary["american_geq_european_call"])
            self.assertLess(summary["no_div_call_diff"], 1.0e-3)
            self.assertAlmostEqual(summary["convergence_ratio"], 4.0, places=5)


if __name__ == "__main__":
    unittest.main()
