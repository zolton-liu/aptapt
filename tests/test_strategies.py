from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qfa_agent.strategies import route_task
from qfa_agent.task import load_task


def make_task(root: Path, *, category: str, instruction: str, task_id: str = "route-test"):
    root.mkdir()
    (root / "instruction.md").write_text(instruction, encoding="utf-8")
    (root / "card.toml").write_text(
        'schema_version="2.0"\n'
        f'[task]\nid="{task_id}"\n'
        f'[metadata]\ncategory="{category}"\ndifficulty="medium"\n'
        '[provenance]\ndata_cutoff=""\n'
        '[contamination]\ncanary_guid=""\n'
        '[agent]\ntimeout_sec=30\n',
        encoding="utf-8",
    )
    return load_task(root)


class StrategyRoutingTests(unittest.TestCase):
    def test_uses_official_card_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(
                Path(temporary) / "task",
                category="fixed-income",
                instruction="Price a bond and calculate DV01.",
            )
            decision = route_task(task)
        self.assertEqual(decision.strategy.category, "fixed-income")
        self.assertEqual(decision.source, "card")
        self.assertEqual(decision.confidence, 1.0)

    def test_maps_legacy_category(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(
                Path(temporary) / "task",
                category="fx-pricing",
                instruction="Compute a forward cross rate.",
            )
            decision = route_task(task)
        self.assertEqual(decision.strategy.category, "fx")
        self.assertEqual(decision.source, "card-alias")

    def test_maps_public_pricing_categories_to_derivatives_policy(self) -> None:
        for category in ("pricing", "derivatives"):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as temporary:
                task = make_task(
                    Path(temporary) / "task",
                    category=category,
                    instruction="Audit option put-call parity.",
                )
                decision = route_task(task)
            self.assertEqual(decision.strategy.category, "derivatives-pricing")
            self.assertEqual(decision.source, "card-alias")

    def test_maps_extended_public_taxonomy_without_keyword_guessing(self) -> None:
        cases = {
            "factor-models": "factor-research",
            "fixed-income-nlp": "nlp-on-finance",
            "execution": "microstructure",
            "extreme-value-theory": "risk-management",
            "cross-currency-rates": "fx",
            "event-driven-analysis, data-processing": "nlp-on-finance",
        }
        for category, expected in cases.items():
            with self.subTest(category=category), tempfile.TemporaryDirectory() as temporary:
                task = make_task(
                    Path(temporary) / "task",
                    category=category,
                    instruction="A deliberately vocabulary-neutral task description.",
                )
                decision = route_task(task)
            self.assertEqual(decision.strategy.category, expected)
            self.assertEqual(decision.source, "card-alias")

    def test_routes_ambiguous_tool_task_from_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(
                Path(temporary) / "task",
                category="tool-using",
                instruction="Adjust prices for a corporate action stock split and dividend.",
            )
            decision = route_task(task)
        self.assertEqual(decision.strategy.category, "backtesting")
        self.assertEqual(decision.source, "instruction-keywords")

    def test_all_official_categories_have_distinct_policy(self) -> None:
        expected = {
            "derivatives-pricing", "fixed-income", "credit", "factor-research",
            "backtesting", "risk-management", "microstructure", "fx",
            "nlp-on-finance", "cross-domain", "software-repair",
        }
        from qfa_agent.strategies import STRATEGIES

        self.assertEqual(set(STRATEGIES), expected)
        self.assertEqual(len({value.architecture for value in STRATEGIES.values()}), len(expected))

    def test_routes_debug_migration_to_template_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(
                Path(temporary) / "task",
                category="debug-migration",
                instruction="Fix the supplied implementation under the current Polars API.",
            )
            decision = route_task(task)
        self.assertEqual(decision.strategy.category, "software-repair")
        self.assertEqual(decision.source, "card-alias")

    def test_specific_kernel_overrides_broad_cross_domain_card(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = make_task(
                Path(temporary) / "task",
                category="cross-domain",
                instruction="Price a compound option with the Geske formula.",
                task_id="t1-compound-option-geske",
            )
            decision = route_task(task)
        self.assertEqual(decision.strategy.category, "derivatives-pricing")
        self.assertEqual(decision.source, "task-family-override")


if __name__ == "__main__":
    unittest.main()
