from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qfa_agent.finance_ops import (
    backward_adjust_ohlcv,
    black_cap_floor_metrics,
    black_scholes_metrics,
    cir_log_likelihood,
    clean_return_panel,
    historical_var_metrics,
    write_cir_calibration_outputs,
    write_asian_option_outputs,
    write_cliquet_outputs,
    write_ohlc_volatility_outputs,
    write_put_call_parity_audit,
)


class FinanceOperatorTests(unittest.TestCase):
    def test_cliquet_writer_uses_forward_start_discounting(self) -> None:
        import json
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prices = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=80),
                    "close": [100.0 * (1.001 ** i) * (1.0 + 0.01 * (i % 3)) for i in range(80)],
                }
            )
            data = root / "prices.csv"
            prices.to_csv(data, index=False)
            bundle = write_cliquet_outputs(data, root / "output")
            self.assertEqual(len(bundle["prices"]), 6)
            self.assertEqual(len(bundle["details"]), 48)
            self.assertIsInstance(bundle["summary"]["cliquet_increases_with_N"], bool)
            summary = json.loads((root / "output" / "summary.json").read_text())
            self.assertEqual(summary["total_forward_starts"], 48)
            for row in bundle["prices"]:
                total = sum(
                    item["forward_start_price"]
                    for item in bundle["details"]
                    if item["T"] == row["T"] and item["N"] == row["N"]
                )
                self.assertAlmostEqual(total, row["cliquet_price"], places=10)

    def test_ohlc_writer_creates_five_artifacts(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            values = [100.0 + 0.2 * index + (index % 3) * 0.1 for index in range(300)]
            frame = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=300),
                "open": values,
                "high": [value * 1.01 for value in values],
                "low": [value * 0.99 for value in values],
                "close": [value * (1.001 if index % 2 else 0.999) for index, value in enumerate(values)],
                "volume": 1000,
            })
            path = root / "prices.csv"
            frame.to_csv(path, index=False)
            bundle = write_ohlc_volatility_outputs(path, root / "output")
            self.assertEqual(bundle["calibration"]["n_days"], 300)
            self.assertEqual(len(bundle["rolling_vol_stats"]), 5)
            self.assertEqual(len(list((root / "output").iterdir())), 5)

    def test_asian_writer_uses_all_prices_and_native_boolean(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=40),
                "close": [100.0 * (1.001 ** index) * (1.0 + 0.002 * (index % 2)) for index in range(40)],
            })
            path = root / "prices.csv"
            frame.to_csv(path, index=False)
            bundle = write_asian_option_outputs(path, root / "output", paths=2000, seed=7)
            self.assertEqual(bundle["calibration"]["n_prices"], 40)
            self.assertEqual(len(bundle["prices"]), 18)
            self.assertIsInstance(bundle["summary"]["curran_better_than_levy"], bool)
    def test_cir_bundle_uses_only_supported_market_tenors(self) -> None:
        import pandas as pd
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dates = pd.date_range("2016-01-01", periods=20, freq="D")
            frame = pd.DataFrame({
                "date": dates,
                "DFF": [3.0 + 0.01 * (index % 4) for index in range(20)],
                "DGS1": 3.1, "DGS2": 3.2, "DGS5": 3.4, "DGS10": 3.6,
            })
            data = root / "rates.csv"
            frame.to_csv(data, index=False)

            class Fit:
                fun = -1.0
                x = [0.5, 0.03, 0.05]

            with patch("scipy.optimize.minimize", return_value=Fit()):
                bundle = write_cir_calibration_outputs(data, root / "output")
            bonds = bundle["bond_prices"]
            self.assertTrue(bonds.loc[bonds["tau"] == 0.25, "market_yield"].isna().all())
            self.assertAlmostEqual(
                float(bonds.loc[bonds["tau"] == 1.0, "market_yield"].iloc[0]), 0.031
            )
            self.assertTrue((root / "output" / "summary.json").exists())

    def test_put_call_parity_writer_creates_consistent_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "quotes.csv").write_text(
                "quote_id,expiry_ts,strike,option_type,bid,ask,quote_age_sec\n"
                "C,2025-02-01T00:00:00Z,100,call,2,2.2,1\n"
                "P,2025-02-01T00:00:00Z,100,put,1.8,2,1\n",
                encoding="utf-8",
            )
            (root / "spot.json").write_text(
                '{"quote_ts":"2025-01-01T00:00:00Z","spot_price":100}', encoding="utf-8"
            )
            (root / "carry.json").write_text(
                '{"risk_free_rate":0.02,"dividend_yield":0.0,"reference_borrow_rate":0.0}',
                encoding="utf-8",
            )
            (root / "rules.json").write_text(
                '{"max_quote_age_sec":10,"violation_threshold":0.5}', encoding="utf-8"
            )
            bundle = write_put_call_parity_audit(
                root / "quotes.csv", root / "spot.json", root / "carry.json",
                root / "rules.json", root / "output",
            )
            self.assertEqual(bundle["results"]["matched_pair_count"], 1)
            self.assertEqual(len(bundle["audit_rows"]), 1)
            self.assertTrue((root / "output" / "solution.json").exists())
    def test_black_cap_floor_parity(self) -> None:
        metrics = black_cap_floor_metrics(
            [0.043, 0.048],
            [0.99, 0.97],
            volatility=0.22,
            strike=0.0475,
            notional=1_000_000,
            period_length=0.25,
        )
        self.assertEqual(len(metrics["caplets"]), 2)
        self.assertEqual(len(metrics["floorlets"]), 2)
        self.assertAlmostEqual(metrics["put_call_parity_error"], 0.0, places=9)

    def test_cir_log_likelihood_is_finite_for_positive_path(self) -> None:
        likelihood = cir_log_likelihood(
            [0.03, 0.0302, 0.0298, 0.0301], 0.5, 0.04, 0.08
        )
        self.assertTrue(__import__("math").isfinite(likelihood))

    def test_backward_adjust_ohlcv_applies_actions_only_to_prior_dates(self) -> None:
        import pandas as pd

        prices = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "close": [100.0, 102.0, 51.0],
                "volume": [10, 20, 40],
            }
        )
        adjusted, summary = backward_adjust_ohlcv(
            prices,
            [
                {"date": "2024-01-02", "type": "dividend", "amount": 2.0},
                {"date": "2024-01-03", "type": "split", "ratio": "2:1"},
            ],
        )
        self.assertLess(adjusted.loc[0, "close_adjusted"], adjusted.loc[0, "close_unadjusted"])
        self.assertEqual(adjusted.loc[2, "close_adjusted"], 51.0)
        self.assertEqual(adjusted.loc[0, "volume_adjusted"], 20)
        self.assertEqual(summary["n_actions_applied"], 2)

    def test_black_scholes_metrics_reconcile_put_call_parity(self) -> None:
        call = black_scholes_metrics(100, 100, 1, 0.05, 0.2, option_type="call")
        put = black_scholes_metrics(100, 100, 1, 0.05, 0.2, option_type="put")
        self.assertAlmostEqual(call["price"], 10.4505835722, places=8)
        self.assertAlmostEqual(
            call["price"] - put["price"], 100 - 100 * __import__("math").exp(-0.05), places=10
        )
        self.assertGreater(call["gamma"], 0)
        self.assertGreater(call["vega"], 0)
        self.assertGreater(call["delta"], 0)
        self.assertLess(put["delta"], 0)

    def test_clean_return_panel_and_var(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "calendar.csv").write_text(
                "date\n2024-01-02\n2024-01-03\n2024-01-04\n2024-01-05\n",
                encoding="utf-8",
            )
            (root / "returns.csv").write_text(
                " date , A , B \n"
                "01/02/2024,0.01,0.03\n"
                "2024-01-02,0.02,0.02\n"
                "03-Jan-24,0.30,0.01\n"
                "2024-01-04,,0.01\n"
                "2024-01-05,-0.04,-0.02\n"
                "2024-01-06,0.01,0.01\n",
                encoding="utf-8",
            )
            clean, report = clean_return_panel(
                root / "returns.csv", root / "calendar.csv", ["A", "B"]
            )
            self.assertEqual(
                report,
                {
                    "n_rows_raw": 6,
                    "n_non_trading_days_removed": 1,
                    "n_duplicates_removed": 1,
                    "n_outliers_removed": 1,
                    "n_missing_rows_removed": 1,
                    "n_rows_after_cleaning": 2,
                },
            )
            metrics = historical_var_metrics(clean, ["A", "B"])
            self.assertEqual(metrics["n_observations_clean"], 2)
            self.assertEqual(metrics["worst_day_date"], "2024-01-05")
            self.assertAlmostEqual(metrics["worst_day_return"], -0.03)


if __name__ == "__main__":
    unittest.main()
