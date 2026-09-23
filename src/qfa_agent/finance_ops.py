"""Deterministic, reusable finance compute operators.

LLMs select parameters and assemble task-specific deliverables; pinned numeric
operations live here so common cleaning and tail-risk conventions do not have to
be reimplemented from scratch on every task.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def black_scholes_metrics(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    *,
    dividend_yield: float = 0.0,
    option_type: str = "call",
) -> dict[str, float]:
    """Return Black--Scholes price and analytic Greeks in rate units.

    ``vega`` and ``rho`` are derivatives for a unit (1.00) move in
    volatility/rate, and ``theta`` is per year. The caller remains responsible
    for any task-requested 1%-point or per-day conversion.
    """

    import math

    s = float(spot)
    k = float(strike)
    t = float(maturity)
    r = float(rate)
    sigma = float(volatility)
    q = float(dividend_yield)
    kind = option_type.strip().lower()
    if kind not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if not all(math.isfinite(value) for value in (s, k, t, r, sigma, q)):
        raise ValueError("Black--Scholes inputs must be finite")
    if s <= 0 or k <= 0 or t < 0 or sigma < 0:
        raise ValueError("spot/strike must be positive and maturity/volatility non-negative")

    normal_cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    if t == 0:
        intrinsic = max(s - k, 0.0) if kind == "call" else max(k - s, 0.0)
        delta = (1.0 if s > k else 0.0) if kind == "call" else (-1.0 if s < k else 0.0)
        return {"price": intrinsic, "delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    discount_r = math.exp(-r * t)
    discount_q = math.exp(-q * t)
    if sigma == 0:
        forward_pv = s * discount_q - k * discount_r
        if kind == "call":
            active = forward_pv > 0
            return {
                "price": max(forward_pv, 0.0),
                "delta": discount_q if active else 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": (q * s * discount_q - r * k * discount_r) if active else 0.0,
                "rho": k * t * discount_r if active else 0.0,
            }
        active = forward_pv < 0
        return {
            "price": max(-forward_pv, 0.0),
            "delta": -discount_q if active else 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": (-q * s * discount_q + r * k * discount_r) if active else 0.0,
            "rho": -k * t * discount_r if active else 0.0,
        }

    root_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r - q + 0.5 * sigma * sigma) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    gamma = discount_q * density / (s * sigma * root_t)
    vega = s * discount_q * density * root_t
    diffusion_theta = -(s * discount_q * density * sigma) / (2.0 * root_t)
    if kind == "call":
        price = s * discount_q * normal_cdf(d1) - k * discount_r * normal_cdf(d2)
        delta = discount_q * normal_cdf(d1)
        theta = diffusion_theta - r * k * discount_r * normal_cdf(d2) + q * s * discount_q * normal_cdf(d1)
        rho = k * t * discount_r * normal_cdf(d2)
    else:
        price = k * discount_r * normal_cdf(-d2) - s * discount_q * normal_cdf(-d1)
        delta = discount_q * (normal_cdf(d1) - 1.0)
        theta = diffusion_theta + r * k * discount_r * normal_cdf(-d2) - q * s * discount_q * normal_cdf(-d1)
        rho = -k * t * discount_r * normal_cdf(-d2)
    return {
        "price": float(price),
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "theta": float(theta),
        "rho": float(rho),
    }


def black_cap_floor_metrics(
    forward_rates: Sequence[float],
    discount_factors: Sequence[float],
    *,
    volatility: float,
    strike: float,
    notional: float,
    period_length: float,
) -> dict[str, object]:
    """Price a cap/floor strip with Black's model and reconcile parity."""

    import math

    forwards = [float(value) for value in forward_rates]
    discounts = [float(value) for value in discount_factors]
    if not forwards or len(forwards) != len(discounts):
        raise ValueError("forward_rates and discount_factors must have equal non-zero length")
    sigma = float(volatility)
    k = float(strike)
    principal = float(notional)
    accrual = float(period_length)
    if sigma < 0 or k <= 0 or principal <= 0 or accrual <= 0:
        raise ValueError("invalid cap/floor parameters")
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    caplets: list[float] = []
    floorlets: list[float] = []
    for index, (forward, discount) in enumerate(zip(forwards, discounts)):
        if forward <= 0 or discount <= 0:
            raise ValueError("forwards and discount factors must be positive")
        fixing_time = index * accrual
        scale = principal * accrual * discount
        if fixing_time == 0 or sigma == 0:
            caplet = scale * max(forward - k, 0.0)
            floorlet = scale * max(k - forward, 0.0)
        else:
            root_t = math.sqrt(fixing_time)
            d1 = (math.log(forward / k) + 0.5 * sigma * sigma * fixing_time) / (sigma * root_t)
            d2 = d1 - sigma * root_t
            caplet = scale * (forward * cdf(d1) - k * cdf(d2))
            floorlet = scale * (k * cdf(-d2) - forward * cdf(-d1))
        caplets.append(float(caplet))
        floorlets.append(float(floorlet))
    cap_price = float(sum(caplets))
    floor_price = float(sum(floorlets))
    swap_value = float(
        principal * accrual * sum(df * (forward - k) for forward, df in zip(forwards, discounts))
    )
    return {
        "cap_price": cap_price,
        "floor_price": floor_price,
        "caplets": caplets,
        "floorlets": floorlets,
        "swap_value": swap_value,
        "put_call_parity_error": cap_price - floor_price - swap_value,
    }


def cir_log_likelihood(
    rates: Sequence[float],
    kappa: float,
    theta: float,
    sigma: float,
    *,
    dt: float = 1.0 / 252.0,
) -> float:
    """Exact CIR transition log-likelihood using the scaled nc-chi-square law."""

    import math
    import numpy as np
    from scipy.stats import ncx2

    values = np.asarray(rates, dtype=float)
    kappa = float(kappa)
    theta = float(theta)
    sigma = float(sigma)
    dt = float(dt)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or min(kappa, theta, sigma, dt) <= 0
    ):
        return float("-inf")
    decay = math.exp(-kappa * dt)
    denominator = sigma * sigma * (1.0 - decay)
    if denominator <= 0 or not math.isfinite(denominator):
        return float("-inf")
    c = 2.0 * kappa / denominator
    degrees = 4.0 * kappa * theta / (sigma * sigma)
    noncentrality = 2.0 * c * values[:-1] * decay
    scaled_next = 2.0 * c * values[1:]
    log_density = math.log(2.0 * c) + ncx2.logpdf(
        scaled_next, degrees, noncentrality
    )
    total = float(np.sum(log_density))
    return total if math.isfinite(total) else float("-inf")


def write_cir_calibration_outputs(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    start_date: str = "2016-01-01",
    random_seed: int = 2026,
    restarts: int = 5,
) -> dict[str, object]:
    """Calibrate CIR and write its standard calibration/curve artifact bundle.

    This is intentionally a task-family operator rather than a table of known
    answers: all statistics, parameters, prices, comparisons, and forwards are
    derived from ``data_path``.  Unsupported market tenors are represented as
    missing values instead of being converted into fabricated column names.
    """

    import json
    import math
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    frame = pd.read_csv(data_path, parse_dates=["date"])
    if "DFF" not in frame:
        raise ValueError("CIR input must contain DFF")
    frame = frame.loc[frame["date"] >= pd.Timestamp(start_date)].copy()
    frame["DFF"] = pd.to_numeric(frame["DFF"], errors="coerce") / 100.0
    frame = frame.dropna(subset=["DFF"]).sort_values("date", kind="stable")
    rates = frame["DFF"].to_numpy(dtype=float)
    if len(rates) < 2 or np.any(rates < 0):
        raise ValueError("CIR calibration requires at least two non-negative rates")

    bounds = ((0.01, 1.0), (0.001, 0.10), (0.001, 0.20))

    def objective(parameters):
        value = cir_log_likelihood(rates, *parameters, dt=1.0 / 252.0)
        return -value if math.isfinite(value) else 1.0e100

    rng = np.random.default_rng(int(random_seed))
    starts = [np.array([0.50, max(0.001, min(0.10, rates.mean())), 0.05])]
    while len(starts) < max(5, int(restarts)):
        starts.append(np.array([rng.uniform(low, high) for low, high in bounds]))
    candidates = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    finite = [item for item in candidates if math.isfinite(float(item.fun))]
    if not finite:
        raise ValueError("CIR calibration did not produce a finite likelihood")
    best = min(finite, key=lambda item: float(item.fun))
    kappa, theta, sigma = (float(value) for value in best.x)
    likelihood = cir_log_likelihood(rates, kappa, theta, sigma, dt=1.0 / 252.0)
    if not math.isfinite(likelihood):
        raise ValueError("CIR calibration likelihood is not finite")

    r_current = float(rates[-1])
    gamma = math.sqrt(kappa * kappa + 2.0 * sigma * sigma)

    def price(tau: float) -> float:
        exponent = math.exp(gamma * tau)
        denominator = (gamma + kappa) * (exponent - 1.0) + 2.0 * gamma
        b_value = 2.0 * (exponent - 1.0) / denominator
        a_value = (
            2.0 * gamma * math.exp((kappa + gamma) * tau / 2.0) / denominator
        ) ** (2.0 * kappa * theta / (sigma * sigma))
        return float(a_value * math.exp(-b_value * r_current))

    maturities = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    prices = [price(tau) for tau in maturities]
    zero_rates = [-math.log(value) for value in prices]
    yields = [zero / tau for zero, tau in zip(zero_rates, maturities)]
    market_columns = {1.0: "DGS1", 2.0: "DGS2", 5.0: "DGS5", 10.0: "DGS10"}
    last = frame.iloc[-1]
    market_yields = [
        (
            float(last[market_columns[tau]]) / 100.0
            if tau in market_columns and pd.notna(last.get(market_columns[tau]))
            else float("nan")
        )
        for tau in maturities
    ]
    yield_diffs = [
        model - market if math.isfinite(market) else float("nan")
        for model, market in zip(yields, market_yields)
    ]
    forward_rows = []
    for left, right, p_left, p_right in zip(
        maturities[:-1], maturities[1:], prices[:-1], prices[1:]
    ):
        forward_rows.append(
            {"tau1": left, "tau2": right, "forward_rate": -(math.log(p_right) - math.log(p_left)) / (right - left)}
        )

    ratio = 2.0 * kappa * theta / (sigma * sigma)
    feller = bool(ratio >= 1.0)
    calibration = {
        "n_obs": int(len(rates)),
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "log_likelihood": float(likelihood),
        "feller_condition_satisfied": feller,
        "feller_ratio": float(ratio),
        "r_current": r_current,
        "r_mean": float(np.mean(rates)),
        "r_std": float(np.std(rates)),
        "r_min": float(np.min(rates)),
        "r_max": float(np.max(rates)),
    }
    bond_frame = pd.DataFrame(
        {
            "tau": maturities,
            "zcb_price": prices,
            "zero_rate": zero_rates,
            "model_yield": yields,
            "market_yield": market_yields,
            "yield_diff": yield_diffs,
        }
    )
    finite_errors = [abs(value) for value in yield_diffs if math.isfinite(value)]
    summary = {
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "r_current": r_current,
        "feller_satisfied": feller,
        "mean_yield_error": float(np.mean(finite_errors)),
        "max_yield_error": float(np.max(finite_errors)),
        "yield_curve_upward_sloping": bool(yields[-1] > yields[0]),
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "calibration.json").write_text(
        json.dumps(calibration, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    bond_frame.to_csv(destination / "bond_prices.csv", index=False)
    pd.DataFrame(forward_rows).to_csv(destination / "forward_rates.csv", index=False)
    (destination / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "calibration": calibration,
        "bond_prices": bond_frame,
        "forward_rates": forward_rows,
        "summary": summary,
    }


def write_put_call_parity_audit(
    quotes_path: str | Path,
    spot_path: str | Path,
    carry_path: str | Path,
    rules_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Clean executable option quotes and write a bid/ask parity audit bundle."""

    import csv
    import json
    import math
    import statistics
    from datetime import datetime

    def load_json(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def timestamp(value: str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def fixed(value: float) -> str:
        return f"{value:.6f}"

    with Path(quotes_path).open(encoding="utf-8", newline="") as handle:
        quotes = list(csv.DictReader(handle))
    spot = load_json(spot_path)
    carry = load_json(carry_path)
    rules = load_json(rules_path)
    exclusions = {"stale": 0, "crossed": 0, "one_sided": 0, "missing_pair": 0}
    clean: list[dict[str, object]] = []
    for quote in quotes:
        bid, ask = float(quote["bid"]), float(quote["ask"])
        if bid <= 0.0 or ask <= 0.0:
            exclusions["one_sided"] += 1
        elif bid >= ask:
            exclusions["crossed"] += 1
        elif float(quote["quote_age_sec"]) > float(rules["max_quote_age_sec"]):
            exclusions["stale"] += 1
        else:
            item: dict[str, object] = dict(quote)
            item.update(strike_value=float(quote["strike"]), bid_value=bid, ask_value=ask)
            clean.append(item)

    grouped: dict[tuple[str, float], dict[str, dict[str, object]]] = {}
    for quote in clean:
        key = (str(quote["expiry_ts"]), float(quote["strike_value"]))
        grouped.setdefault(key, {})[str(quote["option_type"])] = quote
    pairs = []
    for key, sides in grouped.items():
        if "call" in sides and "put" in sides:
            pairs.append((key, sides["call"], sides["put"]))
        else:
            exclusions["missing_pair"] += len(sides)
    pairs.sort(key=lambda item: item[0])
    if not pairs:
        raise ValueError("no clean call/put pairs remain after quote cleaning")

    quote_time = timestamp(str(spot["quote_ts"]))
    spot_price = float(spot["spot_price"])
    risk_free = float(carry["risk_free_rate"])
    dividend = float(carry["dividend_yield"])
    reference_borrow = float(carry["reference_borrow_rate"])
    threshold = float(rules["violation_threshold"])
    audit_rows: list[dict[str, object]] = []
    numeric_rows: list[dict[str, object]] = []
    for (expiry, strike), call, put in pairs:
        years = (timestamp(expiry) - quote_time).total_seconds() / (365.0 * 86400.0)
        discount = math.exp(-risk_free * years)
        forward_bid = strike + (float(call["bid_value"]) - float(put["ask_value"])) / discount
        forward_ask = strike + (float(call["ask_value"]) - float(put["bid_value"])) / discount
        forward_mid = (forward_bid + forward_ask) / 2.0
        theoretical = spot_price * math.exp((risk_free - dividend - reference_borrow) * years)
        implied_borrow = risk_free - dividend - math.log(forward_mid / spot_price) / years
        violation = max(forward_bid - theoretical, theoretical - forward_ask, 0.0)
        if violation > threshold and theoretical < forward_bid:
            kind = "forward_too_high"
        elif violation > threshold and theoretical > forward_ask:
            kind = "forward_too_low"
        else:
            kind = "none"
        numeric = {
            "expiry_ts": expiry, "strike": strike,
            "call_quote_id": str(call["quote_id"]), "put_quote_id": str(put["quote_id"]),
            "forward_bid": forward_bid, "forward_ask": forward_ask,
            "forward_mid": forward_mid, "theoretical": theoretical,
            "implied_borrow": implied_borrow, "violation": violation, "violation_type": kind,
        }
        numeric_rows.append(numeric)
        audit_rows.append({
            "expiry_ts": expiry, "strike": fixed(strike),
            "call_quote_id": call["quote_id"], "put_quote_id": put["quote_id"],
            "time_to_expiry_years": fixed(years),
            "call_bid": fixed(float(call["bid_value"])), "call_ask": fixed(float(call["ask_value"])),
            "put_bid": fixed(float(put["bid_value"])), "put_ask": fixed(float(put["ask_value"])),
            "synthetic_forward_bid": fixed(forward_bid), "synthetic_forward_ask": fixed(forward_ask),
            "synthetic_forward_mid": fixed(forward_mid), "theoretical_forward": fixed(theoretical),
            "implied_borrow_rate": fixed(implied_borrow), "violation_amount": fixed(violation),
            "violation_type": kind,
        })

    violation_rows = [
        {
            "expiry_ts": row["expiry_ts"], "strike": fixed(float(row["strike"])),
            "call_quote_id": row["call_quote_id"], "put_quote_id": row["put_quote_id"],
            "violation_type": row["violation_type"], "violation_amount": fixed(float(row["violation"])),
            "synthetic_forward_bid": fixed(float(row["forward_bid"])),
            "synthetic_forward_ask": fixed(float(row["forward_ask"])),
            "theoretical_forward": fixed(float(row["theoretical"])),
        }
        for row in numeric_rows if float(row["violation"]) > threshold
    ]
    audit_columns = [
        "expiry_ts", "strike", "call_quote_id", "put_quote_id", "time_to_expiry_years",
        "call_bid", "call_ask", "put_bid", "put_ask", "synthetic_forward_bid",
        "synthetic_forward_ask", "synthetic_forward_mid", "theoretical_forward",
        "implied_borrow_rate", "violation_amount", "violation_type",
    ]
    violation_columns = [
        "expiry_ts", "strike", "call_quote_id", "put_quote_id", "violation_type",
        "violation_amount", "synthetic_forward_bid", "synthetic_forward_ask", "theoretical_forward",
    ]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, columns, rows in (
        ("parity_audit.csv", audit_columns, audit_rows),
        ("violations.csv", violation_columns, violation_rows),
    ):
        with (destination / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    worst = max(numeric_rows, key=lambda row: float(row["violation"]))
    pair_ids = [str(row["expiry_ts"]) + "|" + fixed(float(row["strike"])) for row in numeric_rows]
    results = {
        "schema_version": "1.0.0", "task_slug": "option-put-call-parity-forward-audit",
        "quote_ts": str(spot["quote_ts"]), "spot_price": round(spot_price, 6),
        "matched_pair_count": len(numeric_rows), "parity_break_count": len(violation_rows),
        "worst_abs_violation": round(float(worst["violation"]), 6),
        "median_synthetic_forward": round(statistics.median(float(row["forward_mid"]) for row in numeric_rows), 6),
        "median_implied_borrow_rate": round(statistics.median(float(row["implied_borrow"]) for row in numeric_rows), 6),
        "excluded_stale_count": exclusions["stale"], "excluded_crossed_count": exclusions["crossed"],
        "excluded_one_sided_count": exclusions["one_sided"],
        "excluded_missing_pair_count": exclusions["missing_pair"],
    }
    solution = {"intermediates": {
        "exclusion_counts": {"value": exclusions}, "matched_pair_ids": {"value": pair_ids},
        "checkpoint_forwards": {"value": {
            pair_id: round(float(row["forward_mid"]), 6)
            for pair_id, row in zip(pair_ids, numeric_rows)
        }},
        "worst_violation_pair": {"value": str(worst["expiry_ts"]) + "|" + fixed(float(worst["strike"]))},
    }}
    for filename, payload in (("results.json", results), ("solution.json", solution)):
        (destination / filename).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return {"results": results, "audit_rows": audit_rows, "violations": violation_rows, "solution": solution}


def write_ohlc_volatility_outputs(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    annualization: float = 252.0,
) -> dict[str, object]:
    """Compute five OHLC volatility estimators and their rolling diagnostics."""

    import json
    import math
    import numpy as np
    import pandas as pd

    estimators = (
        ("close_to_close", "cc"),
        ("parkinson", "parkinson"),
        ("garman_klass", "garman_klass"),
        ("rogers_satchell", "rogers_satchell"),
        ("yang_zhang", "yang_zhang"),
    )

    def estimate_variances(frame):
        if len(frame) < 3:
            raise ValueError("at least two return observations are required")
        previous_close = frame["close"].to_numpy(dtype=float)[:-1]
        current = frame.iloc[1:]
        opens = current["open"].to_numpy(dtype=float)
        highs = current["high"].to_numpy(dtype=float)
        lows = current["low"].to_numpy(dtype=float)
        closes = current["close"].to_numpy(dtype=float)
        n = len(current)
        close_returns = np.log(closes / previous_close)
        overnight = np.log(opens / previous_close)
        open_close = np.log(closes / opens)
        high_low = np.log(highs / lows)
        rs_terms = np.log(highs / closes) * np.log(highs / opens)
        rs_terms += np.log(lows / closes) * np.log(lows / opens)
        values = {
            "cc": float(np.var(close_returns, ddof=1)),
            "parkinson": float(np.mean(high_low**2) / (4.0 * math.log(2.0))),
            "garman_klass": float(np.mean(
                0.5 * high_low**2
                - (2.0 * math.log(2.0) - 1.0) * open_close**2
            )),
            "rogers_satchell": float(np.mean(rs_terms)),
        }
        k_value = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
        values["yang_zhang"] = float(
            np.var(overnight, ddof=1)
            + k_value * np.var(open_close, ddof=1)
            + (1.0 - k_value) * values["rogers_satchell"]
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("OHLC estimator produced a non-positive/non-finite variance")
        return values

    def rolling_variances(frame, window):
        return [
            estimate_variances(frame.iloc[start : start + window + 1])
            for start in range(len(frame) - window)
        ]

    def annual_vol(value):
        return math.sqrt(value * float(annualization))

    frame = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    numeric = ["open", "high", "low", "close", "volume"]
    if frame[numeric].isna().any().any():
        raise ValueError("OHLCV data contains missing numeric observations")
    positive = bool((frame[["open", "high", "low", "close"]] > 0.0).all().all())
    bounds_valid = bool((
        (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    ).all())
    calibration = {
        "n_days": int(len(frame)),
        "n_returns": int(len(frame) - 1),
        "S0": float(frame["close"].iloc[-1]),
        "ohlc_valid": bool(positive and bounds_valid),
    }
    full_variances = estimate_variances(frame)
    full_vols = {key: annual_vol(value) for key, value in full_variances.items()}
    full_sample = {
        "cc_vol": full_vols["cc"],
        "parkinson_vol": full_vols["parkinson"],
        "garman_klass_vol": full_vols["garman_klass"],
        "rogers_satchell_vol": full_vols["rogers_satchell"],
        "yang_zhang_vol": full_vols["yang_zhang"],
    }
    rolling_21 = rolling_variances(frame, 21)
    variance_series = {
        key: np.asarray([row[key] for row in rolling_21], dtype=float)
        for _, key in estimators
    }
    sampling_variance = {
        key: float(np.var(values, ddof=1)) for key, values in variance_series.items()
    }
    baseline = sampling_variance["cc"]
    efficiencies = {key: baseline / sampling_variance[key] for _, key in estimators}
    rolling_rows = []
    for label, key in estimators:
        vols = np.sqrt(variance_series[key] * float(annualization))
        rolling_rows.append({
            "estimator": label,
            "mean_vol": float(np.mean(vols)),
            "std_vol": float(np.std(vols, ddof=1)),
            "min_vol": float(np.min(vols)),
            "max_vol": float(np.max(vols)),
            "efficiency": float(efficiencies[key]),
        })
    term_rows = []
    for window in (5, 10, 21, 63, 126, 252):
        window_values = rolling_variances(frame, window)
        term_rows.append({
            "window": window,
            "cc_vol": float(np.mean([annual_vol(row["cc"]) for row in window_values])),
            "yz_vol": float(np.mean([annual_vol(row["yang_zhang"]) for row in window_values])),
        })
    range_keys = ("parkinson", "garman_klass", "rogers_satchell", "yang_zhang")
    summary = {
        "n_rolling_windows": len(rolling_21),
        "cc_vol_annual": full_vols["cc"],
        "yz_vol_annual": full_vols["yang_zhang"],
        "parkinson_vol_annual": full_vols["parkinson"],
        "gk_vol_annual": full_vols["garman_klass"],
        "rs_vol_annual": full_vols["rogers_satchell"],
        "yz_efficiency": efficiencies["yang_zhang"],
        "parkinson_efficiency": efficiencies["parkinson"],
        "gk_efficiency": efficiencies["garman_klass"],
        "rs_efficiency": efficiencies["rogers_satchell"],
        "range_estimators_lower_variance": bool(
            all(sampling_variance[key] < baseline for key in range_keys)
        ),
        "yz_most_efficient": bool(
            efficiencies["yang_zhang"] == max(efficiencies.values())
        ),
        "vol_mean_range": [min(full_vols.values()), max(full_vols.values())],
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, payload in (
        ("calibration.json", calibration),
        ("full_sample_vol.json", full_sample),
        ("summary.json", summary),
    ):
        (destination / filename).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    pd.DataFrame(rolling_rows).to_csv(destination / "rolling_vol_stats.csv", index=False)
    pd.DataFrame(term_rows).to_csv(destination / "vol_term_structure.csv", index=False)
    return {
        "calibration": calibration,
        "full_sample_vol": full_sample,
        "rolling_vol_stats": rolling_rows,
        "vol_term_structure": term_rows,
        "summary": summary,
    }


def write_asian_option_outputs(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    rate: float = 0.05,
    paths: int = 100_000,
    seed: int = 2026,
) -> dict[str, object]:
    """Write a discrete-Asian price grid using closed form, moments, and MC.

    Curran's geometric-conditioning approximation is evaluated by deterministic
    Gauss--Hermite quadrature over the conditioning geometric average.
    """

    import json
    import math
    import numpy as np
    import pandas as pd

    frame = pd.read_csv(data_path).sort_values("date", kind="stable").reset_index(drop=True)
    closes = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=float)
    if len(closes) < 3 or np.any(closes <= 0.0):
        raise ValueError("Asian calibration requires at least three positive closes")
    returns = np.diff(np.log(closes))
    return_std = float(np.std(returns, ddof=1))
    sigma = return_std * math.sqrt(252.0)
    spot = float(closes[-1])
    normal_cdf = lambda value: 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def lognormal_call(log_mean, log_variance, strike, maturity):
        standard = math.sqrt(log_variance)
        d2 = (log_mean - math.log(strike)) / standard
        d1 = d2 + standard
        expected = math.exp(log_mean + 0.5 * log_variance)
        return math.exp(-float(rate) * maturity) * (
            expected * normal_cdf(d1) - strike * normal_cdf(d2)
        )

    def analytic_values(maturity, monitoring, strike):
        times = np.arange(1, monitoring + 1, dtype=float) * maturity / monitoring
        covariance = sigma * sigma * np.minimum.outer(times, times)
        log_means = math.log(spot) + (float(rate) - 0.5 * sigma * sigma) * times
        geometric_mean = float(np.mean(log_means))
        geometric_variance = float(np.mean(covariance))
        geometric = lognormal_call(geometric_mean, geometric_variance, strike, maturity)

        first = float(np.mean(spot * np.exp(float(rate) * times)))
        second = float(np.mean(
            spot * spot * np.exp(
                float(rate) * (times[:, None] + times[None, :]) + covariance
            )
        ))
        matched_variance = math.log(second / (first * first))
        matched_mean = math.log(first) - 0.5 * matched_variance
        levy = lognormal_call(matched_mean, matched_variance, strike, maturity)

        cov_with_geometric = covariance.mean(axis=1)
        beta = cov_with_geometric / geometric_variance
        conditional_variance = sigma * sigma * times - cov_with_geometric**2 / geometric_variance
        nodes, weights = np.polynomial.hermite.hermgauss(48)
        conditioned_log_g = geometric_mean + math.sqrt(2.0 * geometric_variance) * nodes
        conditional_arithmetic = np.mean(
            np.exp(
                log_means[:, None]
                + beta[:, None] * (conditioned_log_g[None, :] - geometric_mean)
                + 0.5 * conditional_variance[:, None]
            ),
            axis=0,
        )
        curran = math.exp(-float(rate) * maturity) * float(
            np.sum(weights * np.maximum(conditional_arithmetic - strike, 0.0)) / math.sqrt(math.pi)
        )
        return float(geometric), float(levy), float(curran)

    rng = np.random.default_rng(int(seed))
    rows = []
    for maturity in (0.5, 1.0):
        for monitoring in (4, 12, 52):
            dt = maturity / monitoring
            values = np.full(int(paths), spot, dtype=float)
            arithmetic_sum = np.zeros(int(paths), dtype=float)
            log_sum = np.zeros(int(paths), dtype=float)
            drift = (float(rate) - 0.5 * sigma * sigma) * dt
            diffusion = sigma * math.sqrt(dt)
            for _ in range(monitoring):
                values *= np.exp(drift + diffusion * rng.standard_normal(int(paths)))
                arithmetic_sum += values
                log_sum += np.log(values)
            arithmetic_average = arithmetic_sum / monitoring
            geometric_average = np.exp(log_sum / monitoring)
            discount = math.exp(-float(rate) * maturity)
            for moneyness in (0.95, 1.0, 1.05):
                strike = spot * moneyness
                arithmetic_payoff = discount * np.maximum(arithmetic_average - strike, 0.0)
                geometric_payoff = discount * np.maximum(geometric_average - strike, 0.0)
                geo_exact, levy, curran = analytic_values(maturity, monitoring, strike)
                rows.append({
                    "T": maturity,
                    "n_monitoring": monitoring,
                    "K": strike,
                    "moneyness": moneyness,
                    "geo_exact": geo_exact,
                    "levy_arith": levy,
                    "curran_arith": curran,
                    "mc_arith": float(np.mean(arithmetic_payoff)),
                    "mc_geo": float(np.mean(geometric_payoff)),
                    "mc_std_err": float(np.std(arithmetic_payoff, ddof=1) / math.sqrt(int(paths))),
                })
    result_frame = pd.DataFrame(rows).sort_values(
        ["T", "n_monitoring", "K"], kind="stable"
    ).reset_index(drop=True)
    levy_errors = np.abs(result_frame["levy_arith"] - result_frame["mc_arith"])
    curran_errors = np.abs(result_frame["curran_arith"] - result_frame["mc_arith"])
    summary = {
        "n_rows": int(len(result_frame)),
        "geo_less_than_arith_pct": float(
            100.0 * np.mean(result_frame["geo_exact"] <= result_frame["mc_arith"] + 0.01)
        ),
        "levy_vs_mc_max_error": float(levy_errors.max()),
        "curran_vs_mc_max_error": float(curran_errors.max()),
        "curran_better_than_levy": bool(curran_errors.max() < levy_errors.max()),
    }
    calibration = {
        "n_prices": int(len(closes)),
        "sigma": float(sigma),
        "S0": spot,
        "return_mean": float(np.mean(returns)),
        "return_std": return_std,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(destination / "asian_prices.csv", index=False)
    for filename, payload in (("calibration.json", calibration), ("summary.json", summary)):
        (destination / filename).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return {"calibration": calibration, "prices": result_frame, "summary": summary}


def backward_adjust_ohlcv(
    prices,
    actions: Sequence[dict[str, object]],
    *,
    date_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
):
    """Apply dividends/splits chronologically to prior OHLCV observations.

    Returns ``(adjusted_frame, summary)``. An action on a non-trading day uses
    the nearest prior close to determine a dividend factor, while adjustments
    still apply to observations strictly before the declared action date.
    """

    import pandas as pd

    frame = prices.copy() if isinstance(prices, pd.DataFrame) else pd.read_csv(prices)
    required = {date_column, close_column, volume_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing price columns: {missing}")
    frame = frame.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)
    unadjusted_close = pd.to_numeric(frame[close_column], errors="raise").astype(float)
    unadjusted_volume = pd.to_numeric(frame[volume_column], errors="raise").astype(float)
    adjusted_close = unadjusted_close.copy()
    adjusted_volume = unadjusted_volume.copy()
    cumulative_price = 1.0
    cumulative_volume = 1.0
    dividends = 0.0

    ordered = sorted(actions, key=lambda item: pd.Timestamp(str(item["date"])))
    for action in ordered:
        action_date = pd.Timestamp(str(action["date"]))
        prior_mask = frame[date_column] < action_date
        kind = str(action.get("type", "")).lower()
        if kind == "split":
            left, right = str(action["ratio"]).split(":", 1)
            ratio = float(left) / float(right)
            if ratio <= 0:
                raise ValueError("split ratio must be positive")
            price_factor = 1.0 / ratio
            volume_factor = ratio
            adjusted_close.loc[prior_mask] *= price_factor
            adjusted_volume.loc[prior_mask] *= volume_factor
            cumulative_price *= price_factor
            cumulative_volume *= volume_factor
        elif kind == "dividend":
            amount = float(action["amount"])
            eligible = frame.loc[frame[date_column] <= action_date]
            if eligible.empty:
                raise ValueError(f"no price on or before dividend date {action_date.date()}")
            close_on_ex_date = float(unadjusted_close.loc[eligible.index[-1]])
            price_factor = (close_on_ex_date - amount) / close_on_ex_date
            if price_factor <= 0:
                raise ValueError("dividend adjustment factor must be positive")
            adjusted_close.loc[prior_mask] *= price_factor
            cumulative_price *= price_factor
            dividends += amount
        else:
            raise ValueError(f"unsupported corporate action type: {kind!r}")

    result = pd.DataFrame(
        {
            "date": frame[date_column].dt.strftime("%Y-%m-%d"),
            "close_unadjusted": unadjusted_close,
            "close_adjusted": adjusted_close.round(4),
            "volume_unadjusted": unadjusted_volume.round().astype(int),
            "volume_adjusted": adjusted_volume.round().astype(int),
        }
    )
    summary = {
        "n_actions_applied": int(len(ordered)),
        "cumulative_price_adjustment_factor": round(float(cumulative_price), 8),
        "cumulative_volume_adjustment_factor": round(float(cumulative_volume), 4),
        "first_date_adjusted_close": round(float(adjusted_close.iloc[0]), 4),
        "last_date_unadjusted_close": round(float(unadjusted_close.iloc[-1]), 4),
        "total_dividends_per_original_share": round(float(dividends), 4),
    }
    return result, summary


def _mixed_dates(series):
    import pandas as pd

    try:
        return pd.to_datetime(series, format="mixed", errors="raise")
    except TypeError:  # pandas before format="mixed"
        return series.map(pd.to_datetime)


def clean_return_panel(
    raw_path: str | Path,
    calendar_path: str | Path,
    assets: Sequence[str],
    *,
    date_column: str = "date",
    outlier_abs: float = 0.25,
):
    """Clean a dated return panel in an auditable canonical order.

    Returns ``(clean_frame, report)``. Missing numeric values are preserved
    until the final listwise-deletion step, so they are not miscounted as
    outliers.
    """

    import pandas as pd

    frame = raw_path.copy() if isinstance(raw_path, pd.DataFrame) else pd.read_csv(raw_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    expected = [date_column, *assets]
    missing_columns = [column for column in expected if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"missing return-panel columns: {missing_columns}")
    frame = frame[expected].copy()
    report: dict[str, int] = {"n_rows_raw": int(len(frame))}

    frame[date_column] = _mixed_dates(frame[date_column]).dt.normalize()
    calendar = pd.read_csv(calendar_path)
    calendar.columns = [str(column).strip() for column in calendar.columns]
    if date_column not in calendar.columns:
        raise ValueError(f"calendar is missing {date_column!r}")
    valid_dates = set(_mixed_dates(calendar[date_column]).dt.normalize())

    before = len(frame)
    frame = frame[frame[date_column].isin(valid_dates)].copy()
    report["n_non_trading_days_removed"] = int(before - len(frame))

    before = len(frame)
    frame = frame.drop_duplicates(subset=[date_column], keep="last").copy()
    report["n_duplicates_removed"] = int(before - len(frame))

    for asset in assets:
        frame[asset] = pd.to_numeric(frame[asset], errors="coerce")
    before = len(frame)
    outlier = frame[list(assets)].abs().gt(float(outlier_abs)).any(axis=1)
    frame = frame.loc[~outlier].copy()
    report["n_outliers_removed"] = int(before - len(frame))

    before = len(frame)
    frame = frame.dropna(subset=list(assets)).copy()
    report["n_missing_rows_removed"] = int(before - len(frame))

    frame = frame.sort_values(date_column, kind="stable").reset_index(drop=True)
    report["n_rows_after_cleaning"] = int(len(frame))
    return frame, report


def historical_var_metrics(
    clean_frame,
    assets: Sequence[str],
    *,
    weights: Sequence[float] | None = None,
    date_column: str = "date",
) -> dict[str, object]:
    """Compute daily historical 95%/99% VaR and the worst observation."""

    import numpy as np

    if weights is None:
        weights = [1.0 / len(assets)] * len(assets)
    vector = np.asarray(weights, dtype=float)
    if vector.shape != (len(assets),) or not np.isfinite(vector).all():
        raise ValueError("weights must be one finite value per asset")
    values = clean_frame[list(assets)].to_numpy(dtype=float)
    portfolio = values @ vector
    if not portfolio.size or not np.isfinite(portfolio).all():
        raise ValueError("clean return panel produced no finite portfolio observations")
    worst = int(np.argmin(portfolio))
    return {
        "n_observations_clean": int(portfolio.size),
        "var_95_portfolio": float(-np.percentile(portfolio, 5, method="linear")),
        "var_99_portfolio": float(-np.percentile(portfolio, 1, method="linear")),
        "worst_day_return": float(portfolio[worst]),
        "worst_day_date": clean_frame.iloc[worst][date_column].strftime("%Y-%m-%d"),
    }


def write_cliquet_outputs(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    rate: float = 0.05,
    dividend_yield: float = 0.013,
) -> dict[str, object]:
    """Calibrate SPY volatility and price an ATM forward-start cliquet strip.

    Each reset option has strike equal to spot at its reset date.  Under the
    risk-neutral measure its time-zero value is the unit-maturity Black--Scholes
    ATM factor multiplied by ``S0 * exp(-q * t_start)``.  This is not the same
    as summing ordinary time-zero European calls with increasing maturities.
    """

    import json
    import math

    import pandas as pd
    from scipy.stats import kurtosis, skew

    frame = pd.read_csv(data_path)
    if "close" not in frame.columns:
        raise ValueError("cliquet input must contain a close column")
    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if len(close) < 2 or (close <= 0).any():
        raise ValueError("cliquet calibration needs at least two positive closes")
    log_returns = close.map(math.log).diff().dropna()
    sigma = float(log_returns.std(ddof=1) * math.sqrt(252.0))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("annualized volatility must be positive and finite")
    spot = float(close.iloc[-1])
    r = float(rate)
    q = float(dividend_yield)
    normal_cdf = lambda x: 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    calibration = {
        "n_returns": int(len(log_returns)),
        "return_mean": float(log_returns.mean()),
        "return_std": float(log_returns.std(ddof=1)),
        "return_skewness": float(skew(log_returns)),
        "return_excess_kurtosis": float(kurtosis(log_returns, fisher=True)),
        "annualized_vol": sigma,
        "S0": spot,
    }
    prices: list[dict[str, float | int]] = []
    details: list[dict[str, float | int]] = []
    for maturity in (1.0, 2.0, 3.0):
        european = black_scholes_metrics(
            spot,
            spot,
            maturity,
            r,
            sigma,
            dividend_yield=q,
            option_type="call",
        )["price"]
        for periods in (4, 12):
            dt = maturity / periods
            root_dt = math.sqrt(dt)
            d1 = ((r - q) + 0.5 * sigma * sigma) * root_dt / sigma
            d2 = d1 - sigma * root_dt
            unit_atm = math.exp(-q * dt) * normal_cdf(d1) - math.exp(-r * dt) * normal_cdf(d2)
            forward_values: list[float] = []
            for index in range(periods):
                start = index * dt
                value = spot * math.exp(-q * start) * unit_atm
                forward_values.append(value)
                details.append(
                    {
                        "T": maturity,
                        "N": periods,
                        "period_index": index,
                        "t_start": start,
                        "t_end": (index + 1) * dt,
                        "forward_start_price": value,
                    }
                )
            cliquet = float(sum(forward_values))
            prices.append(
                {
                    "T": maturity,
                    "N": periods,
                    "period_length": dt,
                    "cliquet_price": cliquet,
                    "european_price": european,
                    "ratio_cliquet_over_european": cliquet / european,
                }
            )

    price_frame = pd.DataFrame(prices).sort_values(["T", "N"], kind="stable")
    detail_frame = pd.DataFrame(details).sort_values(
        ["T", "N", "period_index"], kind="stable"
    )
    summary = {
        "n_configurations": int(len(price_frame)),
        "total_forward_starts": int(len(detail_frame)),
        "max_cliquet_price": float(price_frame["cliquet_price"].max()),
        "min_cliquet_price": float(price_frame["cliquet_price"].min()),
        "mean_ratio_cliquet_over_european": float(
            price_frame["ratio_cliquet_over_european"].mean()
        ),
        "cliquet_increases_with_N": bool(
            all(
                float(group.loc[group["N"] == 12, "cliquet_price"].iloc[0])
                > float(group.loc[group["N"] == 4, "cliquet_price"].iloc[0])
                for _, group in price_frame.groupby("T", sort=True)
            )
        ),
    }

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "calibration.json").write_text(
        json.dumps(calibration, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    price_frame.to_csv(target / "cliquet_prices.csv", index=False)
    detail_frame.to_csv(target / "forward_start_details.csv", index=False)
    (target / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "calibration": calibration,
        "prices": prices,
        "details": details,
        "summary": summary,
    }
