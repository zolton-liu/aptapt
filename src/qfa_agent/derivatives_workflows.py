"""Validated implied-volatility and option-replication workflow operators."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _black_scholes_price(
    spot: float, strike: float, maturity: float, rate: float,
    dividend_yield: float, volatility: float, kind: str = "call",
) -> float:
    root_t = math.sqrt(maturity)
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    if kind == "call":
        return discounted_spot * _normal_cdf(d1) - discounted_strike * _normal_cdf(d2)
    return discounted_strike * _normal_cdf(-d2) - discounted_spot * _normal_cdf(-d1)


def _black_scholes_vega(
    spot: float, strike: float, maturity: float, rate: float,
    dividend_yield: float, volatility: float,
) -> float:
    d1 = (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * volatility * volatility) * maturity
    ) / (volatility * math.sqrt(maturity))
    return spot * math.exp(-dividend_yield * maturity) * _normal_pdf(d1) * math.sqrt(maturity)


def write_implied_volatility_approximation_outputs(
    output_dir: str | Path,
    *,
    spot: float = 500.0,
    rate: float = 0.05,
    dividend_yield: float = 0.013,
    volatilities: Sequence[float] = (0.10, 0.15, 0.20, 0.30, 0.40),
    maturities: Sequence[float] = (0.25, 0.50, 1.00),
    moneyness: Sequence[float] = (0.90, 0.95, 1.00, 1.05, 1.10),
) -> dict[str, object]:
    """Compare four closed-form IV estimators with a safeguarded inversion.

    Li and Corrado--Miller--Hallerbach conventions follow the openly
    published QF-Bench task protocol.  Statistics are calculated from rows,
    never stored as task-specific constants.
    """

    import numpy as np
    import pandas as pd
    from scipy.optimize import brentq

    def brenner(price: float, maturity: float, discounted_spot: float) -> float:
        return price / discounted_spot * math.sqrt(2.0 * math.pi / maturity)

    def li_atm(price: float, maturity: float, discounted_spot: float) -> float:
        intermediate = -3.0 * math.sqrt(math.pi) * price / (4.0 * discounted_spot)
        if not -1.0 <= intermediate <= 1.0:
            return math.nan
        angle = 2.0 * math.pi / 3.0 - math.acos(intermediate) / 3.0
        return 4.0 * math.sqrt(2.0 / maturity) * math.cos(angle)

    def li_non_atm(
        price: float, maturity: float, discounted_spot: float, discounted_strike: float,
    ) -> float:
        half_intrinsic = (discounted_spot - discounted_strike) / 2.0
        central = price - half_intrinsic
        discriminant = central * central - (
            (discounted_spot - discounted_strike) ** 2
            * (1.0 + discounted_spot / discounted_strike) / (2.0 * math.pi)
        )
        if discriminant < -1.0e-10:
            return math.nan
        discriminant = max(discriminant, 0.0)
        return (
            math.sqrt(2.0 * math.pi / maturity)
            * (central + math.sqrt(discriminant))
            / (discounted_spot + discounted_strike)
        )

    def corrado_miller_hallerbach(
        price: float, maturity: float, discounted_spot: float, discounted_strike: float,
    ) -> float:
        ratio = discounted_spot / discounted_strike
        alpha = (
            math.sqrt(2.0 * math.pi) / (ratio + 1.0)
            * (2.0 * price / discounted_strike - (ratio - 1.0))
        )
        beta = 0.5 * ((ratio - 1.0) / (ratio + 1.0)) ** 2
        discriminant = alpha * alpha - 8.0 * beta
        if discriminant < -1.0e-10:
            return math.nan
        return (alpha + math.sqrt(max(discriminant, 0.0))) / math.sqrt(maturity)

    def invert(price: float, strike: float, maturity: float) -> tuple[float, bool]:
        sigma = 0.3
        for _ in range(100):
            value = _black_scholes_price(
                spot, strike, maturity, rate, dividend_yield, sigma, "call"
            )
            vega = _black_scholes_vega(
                spot, strike, maturity, rate, dividend_yield, sigma
            )
            if not math.isfinite(vega) or vega <= 1.0e-14:
                break
            candidate = sigma - (value - price) / vega
            if not math.isfinite(candidate) or candidate <= 0.0 or candidate > 5.0:
                break
            if abs(candidate - sigma) < 1.0e-10:
                return float(candidate), True
            sigma = candidate
        # Independent bracketed path handles Newton's deep ITM/OTM failure modes.
        objective = lambda value: _black_scholes_price(
            spot, strike, maturity, rate, dividend_yield, value, "call"
        ) - price
        try:
            root = brentq(objective, 1.0e-8, 5.0, xtol=1.0e-13, rtol=1.0e-14)
        except ValueError:
            return math.nan, False
        return float(root), True

    rows: list[dict[str, float]] = []
    convergence: list[bool] = []
    for true_sigma in (float(value) for value in volatilities):
        for maturity in (float(value) for value in maturities):
            discounted_spot = float(spot) * math.exp(-float(dividend_yield) * maturity)
            for money in (float(value) for value in moneyness):
                strike = float(spot) * money
                discounted_strike = strike * math.exp(-float(rate) * maturity)
                price = _black_scholes_price(
                    float(spot), strike, maturity, float(rate),
                    float(dividend_yield), true_sigma, "call",
                )
                is_atm = abs(money - 1.0) <= 1.0e-9
                sigma_bs = brenner(price, maturity, discounted_spot) if is_atm else math.nan
                sigma_li_atm = li_atm(price, maturity, discounted_spot) if is_atm else math.nan
                sigma_li = li_non_atm(price, maturity, discounted_spot, discounted_strike)
                sigma_cmh = corrado_miller_hallerbach(
                    price, maturity, discounted_spot, discounted_strike
                )
                sigma_newton, converged = invert(price, strike, maturity)
                convergence.append(converged)
                approximations = [sigma_bs, sigma_li_atm, sigma_li, sigma_cmh, sigma_newton]
                errors = [
                    value - true_sigma if math.isfinite(value) else math.nan
                    for value in approximations
                ]
                rows.append({
                    "sigma_true": true_sigma, "T": maturity, "K": strike,
                    "moneyness": money, "bs_price": price,
                    "sigma_bs_approx": sigma_bs, "sigma_li_atm": sigma_li_atm,
                    "sigma_li_nonatm": sigma_li, "sigma_cmh": sigma_cmh,
                    "sigma_newton": sigma_newton,
                    "error_bs": errors[0], "error_li_atm": errors[1],
                    "error_li_nonatm": errors[2], "error_cmh": errors[3],
                    "error_newton": errors[4],
                })

    frame = pd.DataFrame(rows).sort_values(
        ["sigma_true", "T", "moneyness"], kind="stable"
    ).reset_index(drop=True)

    def rmse(column: str) -> float:
        values = frame[column].dropna().to_numpy(float)
        return float(math.sqrt(float(np.mean(values * values))))

    atm = frame["moneyness"].eq(1.0)
    summary = {
        "n_points": int(len(frame)),
        "n_atm_points": int(atm.sum()),
        "rmse_bs_atm": float(math.sqrt(np.nanmean(frame.loc[atm, "error_bs"] ** 2))),
        "rmse_li_atm": float(math.sqrt(np.nanmean(frame.loc[atm, "error_li_atm"] ** 2))),
        "rmse_li_nonatm": rmse("error_li_nonatm"),
        "rmse_cmh": rmse("error_cmh"),
        "rmse_newton": rmse("error_newton"),
        "max_abs_error_li_nonatm": float(frame["error_li_nonatm"].abs().max()),
        "max_abs_error_cmh": float(frame["error_cmh"].abs().max()),
        "mean_abs_error_cmh": float(frame["error_cmh"].abs().mean()),
        "all_newton_converged": bool(all(convergence)),
    }

    expected_rows = len(volatilities) * len(maturities) * len(moneyness)
    if len(frame) != expected_rows or frame["sigma_newton"].notna().sum() != expected_rows:
        raise AssertionError("IV grid is incomplete")
    for method, error in (
        ("sigma_bs_approx", "error_bs"), ("sigma_li_atm", "error_li_atm"),
        ("sigma_li_nonatm", "error_li_nonatm"), ("sigma_cmh", "error_cmh"),
        ("sigma_newton", "error_newton"),
    ):
        valid = frame[method].notna()
        if not np.allclose(
            frame.loc[valid, method] - frame.loc[valid, "sigma_true"],
            frame.loc[valid, error], atol=1.0e-12,
        ):
            raise AssertionError(f"{method} error column does not reconcile")
    repricing_errors = [
        abs(_black_scholes_price(
            float(spot), float(row.K), float(row.T), float(rate),
            float(dividend_yield), float(row.sigma_newton), "call",
        ) - float(row.bs_price))
        for row in frame.itertuples()
    ]
    if max(repricing_errors) >= 1.0e-8 or not summary["all_newton_converged"]:
        raise AssertionError("independent repricing does not confirm numerical IV")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "approximations.csv", index=False, float_format="%.10f")
    (target / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"approximations": frame, "summary": summary}


def write_variance_swap_outputs(
    option_chain_path: str | Path,
    params_path: str | Path,
    output_dir: str | Path,
    *,
    variance_notional: float = 10_000.0,
    scenarios: Sequence[tuple[str, float]] = (
        ("low", 0.18), ("mid", 0.25), ("high", 0.35),
    ),
) -> dict[str, object]:
    """Clean an option chain, replicate variance, and audit all identities."""

    import numpy as np
    import pandas as pd
    from scipy.optimize import brentq

    params = json.loads(Path(params_path).read_text(encoding="utf-8"))
    chain = pd.read_csv(option_chain_path)
    target_expiry = str(params["target_expiry"])
    maturity = float(params["expiries"][target_expiry])
    forward = float(params["forward_prices"][target_expiry])
    spot = float(params["spot"])
    rate = float(params["risk_free_rate"])
    dividend = float(params["dividend_yield"])
    filters = params["filters"]
    expiry_chain = chain.loc[chain["expiry"].astype(str).eq(target_expiry)].copy()
    expiry_chain = expiry_chain.sort_values("strike", kind="stable").reset_index(drop=True)
    if len(expiry_chain) < 3:
        raise ValueError("target expiry has too few strikes")

    expiry_chain["call_mid"] = (expiry_chain["call_bid"] + expiry_chain["call_ask"]) / 2.0
    expiry_chain["put_mid"] = (expiry_chain["put_bid"] + expiry_chain["put_ask"]) / 2.0
    expiry_chain["otm_is_put"] = expiry_chain["strike"] < forward
    expiry_chain["otm_bid"] = np.where(
        expiry_chain["otm_is_put"], expiry_chain["put_bid"], expiry_chain["call_bid"]
    )
    expiry_chain["otm_ask"] = np.where(
        expiry_chain["otm_is_put"], expiry_chain["put_ask"], expiry_chain["call_ask"]
    )
    expiry_chain["otm_mid"] = np.where(
        expiry_chain["otm_is_put"], expiry_chain["put_mid"], expiry_chain["call_mid"]
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_spread = (
            expiry_chain["otm_ask"] - expiry_chain["otm_bid"]
        ) / expiry_chain["otm_mid"]
    keep = expiry_chain["volume"].ge(float(filters["min_volume"]))
    if bool(filters.get("exclude_zero_bid", True)):
        keep &= expiry_chain["otm_bid"].gt(0.0)
    keep &= relative_spread.notna() & relative_spread.le(float(filters["max_relative_spread"]))
    clean = expiry_chain.loc[keep].copy().reset_index(drop=True)
    if len(clean) < 3 or not (clean["strike"].lt(forward).any() and clean["strike"].ge(forward).any()):
        raise ValueError("clean chain does not straddle the supplied forward")

    implied: list[float] = []
    for row in clean.itertuples():
        kind = "put" if bool(row.otm_is_put) else "call"
        objective = lambda sigma, row=row, kind=kind: _black_scholes_price(
            spot, float(row.strike), maturity, rate, dividend, sigma, kind
        ) - float(row.otm_mid)
        implied.append(float(brentq(objective, 1.0e-8, 5.0, xtol=1.0e-9)))
    clean["iv_strike"] = implied

    below = clean.loc[clean["strike"] < forward].iloc[-1]
    above = clean.loc[clean["strike"] >= forward].iloc[0]
    weight = (forward - below["strike"]) / (above["strike"] - below["strike"])
    iv_atm = float((1.0 - weight) * below["iv_strike"] + weight * above["iv_strike"])
    atm_variance = iv_atm * iv_atm

    strikes = clean["strike"].to_numpy(float)
    prices = clean["otm_mid"].to_numpy(float)
    delta_strike = np.empty(len(strikes), dtype=float)
    delta_strike[0] = strikes[1] - strikes[0]
    delta_strike[-1] = strikes[-1] - strikes[-2]
    delta_strike[1:-1] = (strikes[2:] - strikes[:-2]) / 2.0
    fair_variance = float(
        2.0 * math.exp(rate * maturity) / maturity
        * np.sum(delta_strike / (strikes * strikes) * prices)
    )
    if fair_variance <= 0.0:
        raise AssertionError("replicated fair variance is not positive")
    fair_volatility = math.sqrt(fair_variance)
    scenario_rows = [
        {
            "name": str(name), "sigma_RV": float(sigma_rv),
            "realized_variance": float(sigma_rv) ** 2,
            "pnl": float(variance_notional) * (float(sigma_rv) ** 2 - fair_variance),
        }
        for name, sigma_rv in scenarios
    ]
    result = {
        "target_expiry": target_expiry,
        "forward_price": forward,
        "n_strikes_input": int(len(expiry_chain)),
        "n_strikes_kept": int(len(clean)),
        "n_otm_puts_kept": int(clean["otm_is_put"].sum()),
        "n_otm_calls_kept": int((~clean["otm_is_put"]).sum()),
        "fair_variance_strike": fair_variance,
        "fair_vol": fair_volatility,
        "iv_atm_forward": iv_atm,
        "atm_var": atm_variance,
        "variance_risk_premium": fair_variance - atm_variance,
        "fair_vol_minus_atm_iv": fair_volatility - iv_atm,
        "scenarios": scenario_rows,
    }

    if result["n_otm_puts_kept"] + result["n_otm_calls_kept"] != result["n_strikes_kept"]:
        raise AssertionError("OTM side counts do not reconcile")
    if not math.isclose(result["fair_vol"] ** 2, fair_variance, rel_tol=1.0e-12):
        raise AssertionError("fair volatility does not square to fair variance")
    for row in scenario_rows:
        expected = float(variance_notional) * (row["realized_variance"] - fair_variance)
        if not math.isclose(row["pnl"], expected, abs_tol=1.0e-10):
            raise AssertionError("scenario P&L identity failed")
    repricing_error = 0.0
    for row in clean.itertuples():
        kind = "put" if bool(row.otm_is_put) else "call"
        repricing_error = max(repricing_error, abs(
            _black_scholes_price(
                spot, float(row.strike), maturity, rate, dividend,
                float(row.iv_strike), kind,
            ) - float(row.otm_mid)
        ))
    if repricing_error >= 1.0e-7:
        raise AssertionError("implied volatilities fail independent repricing")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "results.json").write_text(
        json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"results": result, "clean_chain": clean}
