"""Route tasks to specialised quantitative-finance solver policies.

Every Track-1 unit shares one container/tool contract. The inner policy changes by
domain: what to inspect, the computational shape, and the independent invariants
that must be checked before finishing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .task import TaskSpec


@dataclass(frozen=True)
class SolverStrategy:
    category: str
    architecture: str
    workflow: tuple[str, ...]
    invariants: tuple[str, ...]
    failure_modes: tuple[str, ...]
    max_inspection_actions: int
    draft_by_step: int
    max_model_calls: int
    max_python_runs: int
    max_pytest_runs: int

    def prompt_block(self) -> str:
        workflow = "\n".join(f"  {i}. {item}" for i, item in enumerate(self.workflow, 1))
        invariants = "\n".join(f"  - {item}" for item in self.invariants)
        failures = "\n".join(f"  - {item}" for item in self.failure_modes)
        return (
            f"Selected category: {self.category}\n"
            f"Solver architecture: {self.architecture}\n"
            f"Domain workflow:\n{workflow}\n"
            f"Independent checks required before finish:\n{invariants}\n"
            f"High-risk failure modes:\n{failures}\n"
            f"Execution policy: at most {self.max_inspection_actions} inspection actions before "
            f"the first executable draft; write that draft by turn {self.draft_by_step}."
        )


@dataclass(frozen=True)
class RoutingDecision:
    strategy: SolverStrategy
    declared_category: str
    source: str
    confidence: float
    matched_terms: tuple[str, ...] = ()


def _strategy(
    category: str,
    architecture: str,
    workflow: tuple[str, ...],
    invariants: tuple[str, ...],
    failure_modes: tuple[str, ...],
    *,
    inspections: int,
    draft: int,
    calls: int,
    python_runs: int,
    pytest_runs: int = 2,
) -> SolverStrategy:
    return SolverStrategy(
        category, architecture, workflow, invariants, failure_modes,
        inspections, draft, calls, python_runs, pytest_runs,
    )


STRATEGIES: dict[str, SolverStrategy] = {
    "derivatives-pricing": _strategy(
        "derivatives-pricing", "analytic/numerical pricer with no-arbitrage audit",
        (
            "Extract contract conventions, units, payoff, boundary cases, and exact output schema.",
            "Choose the simplest valid closed-form, root, lattice, PDE, or Monte Carlo method.",
            "Implement one vectorised solver with explicit expiry and zero-volatility handling.",
            "Run price/Greek or repricing checks independent of the main computation.",
        ),
        ("price bounds, monotonicity and finite values", "product parity when applicable", "root/PDE/calibration residual and Greek sign/unit checks"),
        ("percent versus decimal volatility", "per-year versus per-day theta", "unstable short maturity"),
        inspections=5, draft=6, calls=18, python_runs=8,
    ),
    "fixed-income": _strategy(
        "fixed-income", "cash-flow engine plus curve/root-solving layer",
        (
            "Extract day count, compounding, coupon, settlement, clean/dirty and notional conventions.",
            "Build explicit dated cash flows and one discount/curve abstraction.",
            "Solve yields or curve knots with bracketed roots; never default a missing knot.",
            "Reprice calibration instruments and perturb rates to audit duration/DV01 signs.",
        ),
        ("cash-flow PV reconciles to reported price", "positive discount factors and task-consistent curve behaviour", "price falls with yield and calibration residual is small"),
        ("compounding mismatch", "coupon-grid off-by-one", "clean/dirty or basis-point confusion"),
        inspections=5, draft=6, calls=18, python_runs=8,
    ),
    "credit": _strategy(
        "credit", "state-probability/leg decomposition with calibration reconciliation",
        (
            "Separate recovery, discounting, premium cash flows and protection/default cash flows.",
            "Build survival/default probabilities on one consistent time grid.",
            "Calibrate with positivity constraints or bracketed roots.",
            "Reprice quoted instruments and reconcile component values to totals.",
        ),
        ("survival stays in [0,1] and does not increase", "hazard/default probabilities are non-negative", "par premium and protection legs balance"),
        ("spread/bp conversion", "unconditional versus survival-conditioned flows", "mixed time grids"),
        inspections=5, draft=6, calls=18, python_runs=8,
    ),
    "factor-research": _strategy(
        "factor-research", "point-in-time panel pipeline with portfolio reconciliation",
        (
            "Identify entity/date keys, publication lags, signal window and forward-return window.",
            "Construct signals cross-sectionally per date and form constrained weights.",
            "Separate formation, execution and evaluation; apply costs after turnover.",
            "Reconcile weights, exposures, IC and compounded performance from row outputs.",
        ),
        ("no future observation enters a signal", "neutrality/leverage/turnover constraints hold", "IC is bounded and performance compounds consistently"),
        ("global instead of per-date ranking", "look-ahead from shift direction", "label misalignment"),
        inspections=4, draft=5, calls=18, python_runs=9,
    ),
    "backtesting": _strategy(
        "backtesting", "event-ordered accounting state machine",
        (
            "Define information, signal, execution and valuation times.",
            "Represent cash, holdings, trades, fees and equity explicitly at every step.",
            "Apply corporate actions, fills and rebalances once in chronological order.",
            "Recompute equity and statistics independently from the transaction ledger.",
        ),
        ("equity equals cash plus marked holdings", "terminal wealth equals compounded net returns", "positions obey constraints and costs are non-negative"),
        ("same-close look-ahead", "double-counted split/dividend", "cost applied to wrong notional"),
        inspections=5, draft=6, calls=19, python_runs=10,
    ),
    "risk-management": _strategy(
        "risk-management", "loss-distribution pipeline with tail and attribution audits",
        (
            "Lock return/loss sign, confidence, horizon, notional and cleaning rules.",
            "Construct aligned scenarios and portfolio P&L before tail statistics.",
            "Compute each requested method independently and retain counts/cutoffs.",
            "Backtest exceedances and reconcile component/stress contributions.",
        ),
        ("ES is at least VaR under positive-loss convention", "risk does not fall with confidence", "components reconcile and all results are finite"),
        ("left/right-tail sign reversal", "quantile interpolation mismatch", "cleaning after returns"),
        inspections=4, draft=5, calls=18, python_runs=9,
    ),
    "microstructure": _strategy(
        "microstructure", "timestamped order/trade state machine with conservation checks",
        (
            "Establish ordering, side convention, quote validity and timestamp tie breaking.",
            "Process events once while retaining inventory, remaining quantity, cash and costs.",
            "Compute spread/impact metrics from matched contemporaneous states.",
            "Reconcile executed quantity and cash flow to the event ledger.",
        ),
        ("bid does not exceed ask", "executed plus remaining equals target", "spreads/costs are non-negative and liquidation respects direction"),
        ("trade-side sign reversal", "N versus N+1 intervals", "joining a trade to a future quote"),
        inspections=5, draft=6, calls=19, python_runs=10,
    ),
    "fx": _strategy(
        "fx", "quote-normalisation and no-arbitrage graph/pricer",
        (
            "Write down base/quote orientation and whether bid or ask is required.",
            "Normalise tenors, day counts, compounding and domestic/foreign rate roles.",
            "Compute forwards, crosses or option values without losing quote direction.",
            "Reconstruct direct rates and both arbitrage directions independently.",
        ),
        ("covered-interest parity", "reciprocal/cross-rate consistency including bid/ask", "requested conversion or option parity identity"),
        ("base/quote inversion", "domestic/foreign rate swap", "mid prices used for executable arbitrage"),
        inspections=4, draft=5, calls=16, python_runs=7,
    ),
    "nlp-on-finance": _strategy(
        "nlp-on-finance", "deterministic document parser followed by bounded aggregation",
        (
            "Inspect document schema, labels, dates, speakers and numeric units.",
            "Prefer deterministic parsing/rules for fixed corpora; retain evidence identifiers.",
            "Aggregate after row extraction and enforce closed labels/ranges.",
            "Check publication time against each downstream measurement window.",
        ),
        ("labels and scores stay in declared ranges", "counts and aggregates reconcile", "documents precede prediction/return windows"),
        ("boilerplate dominating scores", "million/billion unit loss", "mixing analyst and management text"),
        inspections=6, draft=7, calls=18, python_runs=7,
    ),
    "cross-domain": _strategy(
        "cross-domain", "decomposed multi-stage DAG with typed intermediate contracts",
        (
            "Split the instruction into domain subproblems and define interface units/schema.",
            "Implement and check each subproblem independently before aggregation.",
            "Persist intermediate tables in scratch for separate repair.",
            "Aggregate after unit conversion and check one invariant per domain.",
        ),
        ("each subproblem passes a domain invariant", "units, keys and dates agree at joins", "final totals reconcile to components"),
        ("solving one domain only", "unit mismatch across stages", "checking only the final aggregate"),
        inspections=3, draft=4, calls=18, python_runs=9, pytest_runs=3,
    ),
    "software-repair": _strategy(
        "software-repair", "template-first compatibility repair with API/schema regression checks",
        (
            "Copy the supplied implementation to scratch/solve.py before inspecting unrelated data.",
            "Run once to capture the concrete compatibility failure and repair the narrowest call site.",
            "Preserve the original computation, ordering and output contract; do not redesign the pipeline.",
            "Run the repaired program and a focused regression check before validation.",
        ),
        (
            "the supplied input fixture executes end-to-end",
            "declared output schema and ordering are unchanged",
            "deprecated API names are absent from the repaired implementation",
        ),
        (
            "rewriting working business logic",
            "changing schema while fixing an API",
            "patching generated output instead of source",
        ),
        inspections=2, draft=3, calls=14, python_runs=6, pytest_runs=2,
    ),
}


_ALIASES = {
    "credit-analysis": "credit",
    "dependence-modeling": "risk-management",
    # Current public cards use these shorter labels.  Treating them as unknown
    # made pricing tasks compete in a keyword tie and sometimes routed them to
    # factor research merely because that key sorts later alphabetically.
    "derivatives": "derivatives-pricing",
    "fx-pricing": "fx",
    "pricing": "derivatives-pricing",
    "strategy": "factor-research",
    "volatility-modeling": "risk-management",
    "interest-rate-derivatives": "derivatives-pricing",
    # Public cards also use these more specific taxonomy labels.  Encoding
    # their intended policy avoids unstable keyword ties and gives the same
    # task the same architecture when wording changes slightly.
    "cross-asset-analysis": "factor-research",
    "cross-currency-rates": "fx",
    "cross-sectional-strategies": "factor-research",
    "derivatives_pricing": "derivatives-pricing",
    "execution": "microstructure",
    "extreme-value-theory": "risk-management",
    "factor-models": "factor-research",
    "fixed-income-nlp": "nlp-on-finance",
    "fx-strategy": "fx",
    "performance-attribution": "factor-research",
    "portfolio-analysis": "factor-research",
    "predictive-alpha-modeling": "factor-research",
    "risk-modeling": "risk-management",
    "statistical-analysis": "factor-research",
    "stochastic-processes": "risk-management",
    "event-driven-analysis, data-processing": "nlp-on-finance",
    "debug-migration": "software-repair",
}

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "derivatives-pricing": ("option", "implied vol", "black-scholes", "kirk", "margrabe", "greek"),
    "fixed-income": ("bond", "yield curve", "duration", "dv01", "cir", "coupon", "zero-coupon"),
    "credit": ("credit spread", "cds", "hazard", "recovery", "survival probability", "default"),
    "factor-research": ("factor", "pca", "information coefficient", "long-short", "portfolio weight"),
    "backtesting": ("backtest", "corporate action", "split", "transaction cost", "trade ledger"),
    "risk-management": ("value at risk", "var", "expected shortfall", "stress", "copula", "rank correlation"),
    "microstructure": ("order book", "bid-ask", "vwap", "execution", "almgren", "vpin"),
    "fx": ("fx", "foreign exchange", "cross rate", "currency", "covered interest parity"),
    "nlp-on-finance": ("transcript", "sentiment", "10-k", "risk factor extraction", "text classification"),
}


def route_task(task: TaskSpec) -> RoutingDecision:
    """Prefer the card category; handle legacy cards and ambiguous tool tasks safely."""

    declared = task.category.strip().lower()
    text = f"{task.task_id}\n{task.safe_instruction}".lower()

    # Some public cards intentionally use a broad cross-domain/statistical
    # label even though the computational kernel is unambiguous.  Route those
    # kernels to the specialist policy; this keeps the card metadata visible
    # while avoiding a much larger generic DAG budget.
    family_overrides: tuple[tuple[tuple[str, ...], str], ...] = (
        (("compound option", "geske"), "derivatives-pricing"),
        (("barone-adesi",), "derivatives-pricing"),
        (("dupire", "local vol"), "derivatives-pricing"),
        (("structured note",), "derivatives-pricing"),
        (("yield curve", "pca"), "fixed-income"),
        (("limit order book",), "microstructure"),
        (("lob-pc",), "microstructure"),
        (("sec 10-k",), "nlp-on-finance"),
        (("realized volatility estimator",), "risk-management"),
        (("first-passage",), "risk-management"),
    )
    if declared in {"cross-domain", "statistical-analysis", "predictive-alpha-modeling"}:
        for required_terms, target in family_overrides:
            if all(term in text for term in required_terms):
                return RoutingDecision(
                    STRATEGIES[target],
                    declared,
                    "task-family-override",
                    0.98,
                    required_terms,
                )
    if declared in STRATEGIES:
        return RoutingDecision(STRATEGIES[declared], declared, "card", 1.0)
    if declared in _ALIASES:
        target = _ALIASES[declared]
        return RoutingDecision(STRATEGIES[target], declared, "card-alias", 0.95)

    scores: list[tuple[int, str, tuple[str, ...]]] = []
    for category, terms in _KEYWORDS.items():
        matched = tuple(term for term in terms if term in text)
        scores.append((len(matched), category, matched))
    scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_score, best_category, matched = scores[0]
    if best_score:
        return RoutingDecision(
            STRATEGIES[best_category], declared, "instruction-keywords",
            min(0.9, 0.55 + 0.1 * best_score), matched,
        )
    return RoutingDecision(STRATEGIES["cross-domain"], declared, "safe-fallback", 0.25)
