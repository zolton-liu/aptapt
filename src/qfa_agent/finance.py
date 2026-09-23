"""Task-sensitive finance review hints; these are prompts, not global truth."""

from __future__ import annotations


_HINTS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("return", "backtest", "momentum", "signal"), "Check signal/execution lag, adjusted prices, first-period handling, compounding, and look-ahead leakage."),
    (("portfolio", "weight", "allocation", "risk parity"), "Check label alignment, finite weights, the task-specific sum/leverage constraint, cash, and transaction costs."),
    (("option", "black-scholes", "greek", "volatility"), "Check units and signs, expiry/zero-vol limits, no-arbitrage bounds, put-call relations, and numerical stability."),
    (("covariance", "correlation", "pca", "factor"), "Check symmetry, dimensions/labels, positive-semidefinite tolerance, sign indeterminacy, and train/test separation."),
    (("var", "cvar", "expected shortfall", "tail"), "Check loss-vs-return sign conventions, quantile direction, confidence level, monotonicity, and finite edge cases."),
    (("yield", "bond", "discount", "curve", "swap"), "Check cash-flow timing, compounding convention, day-count/unit rules, positive discount factors, and monotonic price/yield behavior."),
    (("probability", "distribution", "copula", "simulation"), "Check probabilities and finite values, deterministic seeding, marginal preservation, and sample-shape conventions."),
    (("pnl", "execution", "trade", "transaction cost"), "Check accounting identity (equity = cash + holdings value), non-negative costs, fills, position limits, and forced close rules."),
    (
        ("bootstrap", "zero-coupon", "par coupon"),
        "For curve bootstrapping, sum the present values of earlier coupon cash flows; never multiply discount factors. Keep maturity keys numeric during all calculations and convert them to output strings only during serialization. Never use dict.get(time, 0) for a missing discount factor: a zero default silently removes a coupon cash flow. When a new maturity T follows a gap, intermediate coupon dates between the previous knot and T depend on the still-unknown z(T): evaluate their zero rates by linear interpolation between the previous known knot and a candidate z(T), then solve par_price(candidate_z) - 1 = 0 with a robust one-dimensional root finder or bisection. A one-shot closed-form terminal-DF formula is valid only when every earlier coupon-date discount factor is already known independently of z(T). After all knots are solved, obtain z(T-1) for forward rates from the final interpolated curve when T-1 is not an output maturity. Before finishing, independently reprice every input par bond to 1 within a tight tolerance and check positive decreasing discount factors.",
    ),
    (
        ("bootstrap", "zero-coupon", "par coupon"),
        "Reusable algorithm shape (adapt names and output schema; never hardcode values): maintain `known_z: dict[float, float]`. For each `(T, c)`, define `z_at(t, candidate_z)` by linear interpolation over the sorted numeric knots `[*known_z, T]` with values `[*known_z.values(), candidate_z]`; define `df_at(t, candidate_z) = exp(-z_at(t, candidate_z) * t)`; define `residual(candidate_z) = (c/freq) * sum(df_at(k/freq, candidate_z) for k in range(1, int(T*freq))) + (1+c/freq) * exp(-candidate_z*T) - 1`; solve `residual(z)=0` using scipy.optimize.brentq with a wide finite bracket (or robust bisection), then store `known_z[float(T)] = z`. After all knots exist, interpolate the final numeric curve for any T-1 needed by forwards, compute each requested `df(T)=exp(-z(T)*T)`, reprice every input bond from the final curve, assert the maximum absolute par-pricing error is small, and only then serialize rounded string keys.",
    ),
)


def finance_hints(instruction: str) -> tuple[str, ...]:
    lowered = instruction.lower()
    selected = [hint for keywords, hint in _HINTS if any(word in lowered for word in keywords)]
    if not selected:
        selected.append("Derive invariants from the instruction; do not impose a universal finance convention when the task specifies its own.")
    selected.append("Test NaN/Inf, empty or short inputs, dtype/shape/order, units, rounding, and deterministic behavior where applicable.")
    return tuple(dict.fromkeys(selected))
