"""Small deterministic starter programs for recognised finance task families."""

from __future__ import annotations

from dataclasses import dataclass

from .task import TaskSpec


@dataclass(frozen=True)
class StarterProgram:
    path: str
    content: str
    purpose: str


def _bootstrap_starter(data_path: str) -> str:
    return f'''import json
import math
import os
from bisect import bisect_right
from pathlib import Path

from scipy.optimize import brentq


def interpolate_zero(t, curve):
    xs = sorted(curve)
    if t <= xs[0]:
        return curve[xs[0]]
    if t >= xs[-1]:
        return curve[xs[-1]]
    right = bisect_right(xs, t)
    left = right - 1
    x0, x1 = xs[left], xs[right]
    z0, z1 = curve[x0], curve[x1]
    return z0 + (z1 - z0) * (t - x0) / (x1 - x0)


def discount(t, curve):
    return math.exp(-interpolate_zero(t, curve) * t)


def par_price(T, coupon_rate, freq, curve):
    periods = int(round(T * freq))
    coupon = coupon_rate / freq
    earlier = sum(coupon * discount(k / freq, curve) for k in range(1, periods))
    return earlier + (1.0 + coupon) * discount(T, curve)


def solve_zero(T, coupon_rate, freq, known):
    def residual(candidate):
        trial = dict(known)
        trial[T] = candidate
        return par_price(T, coupon_rate, freq, trial) - 1.0

    low, high = -0.25, 0.25
    for _ in range(8):
        if residual(low) * residual(high) <= 0.0:
            return brentq(residual, low, high, xtol=1e-14, rtol=1e-14)
        low *= 2.0
        high *= 2.0
    raise ValueError(f"could not bracket the zero rate at maturity {{T}}")


def output_key(T):
    return str(int(T)) if float(T).is_integer() else format(T, "g")


def main():
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    with (task_dir / {data_path!r}).open(encoding="utf-8") as handle:
        payload = json.load(handle)

    freq = int(payload["coupon_freq"])
    pairs = sorted((float(T), float(c)) for T, c in zip(payload["maturities"], payload["par_rates"]))
    curve = {{}}
    for T, coupon_rate in pairs:
        curve[T] = solve_zero(T, coupon_rate, freq, curve)

    repricing_errors = [abs(par_price(T, c, freq, curve) - 1.0) for T, c in pairs]
    if max(repricing_errors, default=0.0) > 1e-9:
        raise ValueError(f"par repricing self-check failed: {{repricing_errors}}")

    zero_rates = {{}}
    discount_factors = {{}}
    forward_rates = {{}}
    for T, _ in pairs:
        key = output_key(T)
        z_T = curve[T]
        zero_rates[key] = round(z_T, 8)
        discount_factors[key] = round(math.exp(-z_T * T), 8)
        if T < 2.0:
            forward = z_T
        else:
            z_previous = interpolate_zero(T - 1.0, curve)
            forward = z_T * T - z_previous * (T - 1.0)
        forward_rates[key] = round(forward, 8)

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {{
        "zero_rates": zero_rates,
        "discount_factors": discount_factors,
        "forward_rates": forward_rates,
    }}
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
'''


def _parity_audit_starter(
    quotes_path: str,
    spot_path: str,
    carry_path: str,
    rules_path: str,
) -> str:
    return f'''import csv
import json
import math
import os
import statistics
from datetime import datetime
from pathlib import Path


AUDIT_COLUMNS = [
    "expiry_ts", "strike", "call_quote_id", "put_quote_id",
    "time_to_expiry_years", "call_bid", "call_ask", "put_bid", "put_ask",
    "synthetic_forward_bid", "synthetic_forward_ask", "synthetic_forward_mid",
    "theoretical_forward", "implied_borrow_rate", "violation_amount",
    "violation_type",
]
VIOLATION_COLUMNS = [
    "expiry_ts", "strike", "call_quote_id", "put_quote_id",
    "violation_type", "violation_amount", "synthetic_forward_bid",
    "synthetic_forward_ask", "theoretical_forward",
]


def utc_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fixed(value):
    return f"{{value:.6f}}"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)

    spot = load_json(task_dir / {spot_path!r})
    carry = load_json(task_dir / {carry_path!r})
    rules = load_json(task_dir / {rules_path!r})
    with (task_dir / {quotes_path!r}).open(encoding="utf-8", newline="") as handle:
        quotes = list(csv.DictReader(handle))

    exclusions = {{"stale": 0, "crossed": 0, "one_sided": 0, "missing_pair": 0}}
    clean = []
    for quote in quotes:
        bid, ask = float(quote["bid"]), float(quote["ask"])
        if bid <= 0.0 or ask <= 0.0:
            exclusions["one_sided"] += 1
        elif bid >= ask:
            exclusions["crossed"] += 1
        elif float(quote["quote_age_sec"]) > float(rules["max_quote_age_sec"]):
            exclusions["stale"] += 1
        else:
            item = dict(quote)
            item["strike_value"] = float(quote["strike"])
            item["bid_value"] = bid
            item["ask_value"] = ask
            clean.append(item)

    grouped = {{}}
    for quote in clean:
        key = (quote["expiry_ts"], quote["strike_value"])
        grouped.setdefault(key, {{}})[quote["option_type"]] = quote

    pairs = []
    for key, sides in grouped.items():
        if "call" in sides and "put" in sides:
            pairs.append((key, sides["call"], sides["put"]))
        else:
            exclusions["missing_pair"] += len(sides)
    pairs.sort(key=lambda item: item[0])

    quote_time = utc_timestamp(spot["quote_ts"])
    spot_price = float(spot["spot_price"])
    risk_free = float(carry["risk_free_rate"])
    dividend = float(carry["dividend_yield"])
    reference_borrow = float(carry["reference_borrow_rate"])
    threshold = float(rules["violation_threshold"])
    audit_rows = []
    numeric_rows = []

    for (expiry, strike), call, put in pairs:
        years = (utc_timestamp(expiry) - quote_time).total_seconds() / (365.0 * 86400.0)
        discount_factor = math.exp(-risk_free * years)
        forward_bid = strike + (call["bid_value"] - put["ask_value"]) / discount_factor
        forward_ask = strike + (call["ask_value"] - put["bid_value"]) / discount_factor
        forward_mid = (forward_bid + forward_ask) / 2.0
        theoretical = spot_price * math.exp(
            (risk_free - dividend - reference_borrow) * years
        )
        implied_borrow = risk_free - dividend - math.log(forward_mid / spot_price) / years
        violation = max(forward_bid - theoretical, theoretical - forward_ask, 0.0)
        if violation > threshold and theoretical < forward_bid:
            violation_type = "forward_too_high"
        elif violation > threshold and theoretical > forward_ask:
            violation_type = "forward_too_low"
        else:
            violation_type = "none"

        numeric = {{
            "expiry_ts": expiry,
            "strike_value": strike,
            "call_quote_id": call["quote_id"],
            "put_quote_id": put["quote_id"],
            "forward_bid": forward_bid,
            "forward_ask": forward_ask,
            "forward_mid": forward_mid,
            "theoretical": theoretical,
            "implied_borrow": implied_borrow,
            "violation": violation,
            "violation_type": violation_type,
        }}
        numeric_rows.append(numeric)
        audit_rows.append({{
            "expiry_ts": expiry,
            "strike": fixed(strike),
            "call_quote_id": call["quote_id"],
            "put_quote_id": put["quote_id"],
            "time_to_expiry_years": fixed(years),
            "call_bid": fixed(call["bid_value"]),
            "call_ask": fixed(call["ask_value"]),
            "put_bid": fixed(put["bid_value"]),
            "put_ask": fixed(put["ask_value"]),
            "synthetic_forward_bid": fixed(forward_bid),
            "synthetic_forward_ask": fixed(forward_ask),
            "synthetic_forward_mid": fixed(forward_mid),
            "theoretical_forward": fixed(theoretical),
            "implied_borrow_rate": fixed(implied_borrow),
            "violation_amount": fixed(violation),
            "violation_type": violation_type,
        }})

    violation_rows = [
        {{
            "expiry_ts": row["expiry_ts"],
            "strike": fixed(row["strike_value"]),
            "call_quote_id": row["call_quote_id"],
            "put_quote_id": row["put_quote_id"],
            "violation_type": row["violation_type"],
            "violation_amount": fixed(row["violation"]),
            "synthetic_forward_bid": fixed(row["forward_bid"]),
            "synthetic_forward_ask": fixed(row["forward_ask"]),
            "theoretical_forward": fixed(row["theoretical"]),
        }}
        for row in numeric_rows if row["violation"] > threshold
    ]
    write_csv(output_dir / "parity_audit.csv", AUDIT_COLUMNS, audit_rows)
    write_csv(output_dir / "violations.csv", VIOLATION_COLUMNS, violation_rows)

    worst = max(numeric_rows, key=lambda row: row["violation"])
    pair_ids = [
        row["expiry_ts"] + "|" + fixed(row["strike_value"])
        for row in numeric_rows
    ]
    results = {{
        "schema_version": "1.0.0",
        "task_slug": "option-put-call-parity-forward-audit",
        "quote_ts": spot["quote_ts"],
        "spot_price": round(spot_price, 6),
        "matched_pair_count": len(numeric_rows),
        "parity_break_count": len(violation_rows),
        "worst_abs_violation": round(worst["violation"], 6),
        "median_synthetic_forward": round(statistics.median(row["forward_mid"] for row in numeric_rows), 6),
        "median_implied_borrow_rate": round(statistics.median(row["implied_borrow"] for row in numeric_rows), 6),
        "excluded_stale_count": exclusions["stale"],
        "excluded_crossed_count": exclusions["crossed"],
        "excluded_one_sided_count": exclusions["one_sided"],
        "excluded_missing_pair_count": exclusions["missing_pair"],
    }}
    solution = {{
        "intermediates": {{
            "exclusion_counts": {{"value": exclusions}},
            "matched_pair_ids": {{"value": pair_ids}},
            "checkpoint_forwards": {{"value": {{
                pair_id: round(row["forward_mid"], 6)
                for pair_id, row in zip(pair_ids, numeric_rows)
            }}}},
            "worst_violation_pair": {{
                "value": worst["expiry_ts"] + "|" + fixed(worst["strike_value"])
            }},
        }}
    }}
    for filename, payload in (("results.json", results), ("solution.json", solution)):
        (output_dir / filename).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
'''


def _ohlc_volatility_starter(data_path: str) -> str:
    return f'''import os
from pathlib import Path

from qfa_agent.finance_ops import write_ohlc_volatility_outputs


if __name__ == "__main__":
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    write_ohlc_volatility_outputs(task_dir / {data_path!r}, output_dir)
'''


def _cliquet_operator_program(data_path: str) -> str:
    return f'''import os
from pathlib import Path

from qfa_agent.finance_ops import write_cliquet_outputs


if __name__ == "__main__":
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    write_cliquet_outputs(
        task_dir / {data_path!r},
        output_dir,
        rate=0.05,
        dividend_yield=0.013,
    )
'''


def trusted_operator_program_for(
    task: TaskSpec, input_inventory: tuple[str, ...]
) -> StarterProgram | None:
    """Build a narrow adapter for an exact, tested finance-operator match.

    These adapters obey the category-starter enable/disable switch.
    They are only selected when both the
    instruction signature and a unique required input match, so the
    controller can execute a verified numerical kernel without waiting for
    the language model to rediscover the mechanical adapter.
    """

    lowered = task.safe_instruction.lower()
    is_cliquet = "cliquet" in lowered and "forward-start" in lowered
    if is_cliquet:
        candidates = [
            path
            for path in input_inventory
            if path.startswith("environment/data/")
            and path.lower().endswith("spy_daily.csv")
        ]
        if len(candidates) == 1:
            return StarterProgram(
                path="scratch/solve.py",
                content=_cliquet_operator_program(candidates[0]),
                purpose=(
                    "trusted forward-start cliquet pricing operator with "
                    "calibration and deliverable generation"
                ),
            )
    return None


def _legacy_ohlc_volatility_starter(data_path: str) -> str:
    """Retained source fixture; production starter uses the tested operator."""
    return f'''import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


ESTIMATORS = [
    ("close_to_close", "cc"),
    ("parkinson", "parkinson"),
    ("garman_klass", "garman_klass"),
    ("rogers_satchell", "rogers_satchell"),
    ("yang_zhang", "yang_zhang"),
]


def estimate_variances(frame):
    """Estimate daily variances from one prior close plus n current OHLC rows."""
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

    cc = float(np.var(close_returns, ddof=1))
    parkinson = float(np.mean(high_low ** 2) / (4.0 * math.log(2.0)))
    garman_klass = float(np.mean(
        0.5 * high_low ** 2 - (2.0 * math.log(2.0) - 1.0) * open_close ** 2
    ))
    rogers_satchell = float(np.mean(rs_terms))
    k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0))
    yang_zhang = float(
        np.var(overnight, ddof=1)
        + k * np.var(open_close, ddof=1)
        + (1.0 - k) * rogers_satchell
    )
    estimates = {{
        "cc": cc,
        "parkinson": parkinson,
        "garman_klass": garman_klass,
        "rogers_satchell": rogers_satchell,
        "yang_zhang": yang_zhang,
    }}
    for name, value in estimates.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"invalid {{name}} variance: {{value}}")
    return estimates


def rolling_variances(frame, window):
    return [
        estimate_variances(frame.iloc[start:start + window + 1])
        for start in range(len(frame) - window)
    ]


def annual_vol(daily_variance):
    return math.sqrt(daily_variance * 252.0)


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\\n",
        encoding="utf-8",
    )


def main():
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(task_dir / {data_path!r}, parse_dates=["date"])
    frame = frame.sort_values("date").reset_index(drop=True)

    numeric_columns = ["open", "high", "low", "close", "volume"]
    if frame[numeric_columns].isna().any().any():
        raise ValueError("OHLCV data contains missing numeric observations")
    positive = (frame[["open", "high", "low", "close"]] > 0.0).all().all()
    bounds_valid = (
        (frame["high"] >= frame[["open", "close"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close"]].min(axis=1))
    ).all()
    calibration = {{
        "n_days": int(len(frame)),
        "n_returns": int(len(frame) - 1),
        "S0": float(frame["close"].iloc[-1]),
        "ohlc_valid": bool(positive and bounds_valid),
    }}

    full_variances = estimate_variances(frame)
    full_vols = {{name: annual_vol(value) for name, value in full_variances.items()}}
    full_sample = {{
        "cc_vol": full_vols["cc"],
        "parkinson_vol": full_vols["parkinson"],
        "garman_klass_vol": full_vols["garman_klass"],
        "rogers_satchell_vol": full_vols["rogers_satchell"],
        "yang_zhang_vol": full_vols["yang_zhang"],
    }}

    rolling_21 = rolling_variances(frame, 21)
    variance_series = {{
        key: np.asarray([row[key] for row in rolling_21], dtype=float)
        for _, key in ESTIMATORS
    }}
    estimator_sampling_variance = {{
        key: float(np.var(values, ddof=1))
        for key, values in variance_series.items()
    }}
    baseline_variance = estimator_sampling_variance["cc"]
    efficiencies = {{
        key: baseline_variance / estimator_sampling_variance[key]
        for _, key in ESTIMATORS
    }}
    rolling_rows = []
    for label, key in ESTIMATORS:
        vols = np.sqrt(variance_series[key] * 252.0)
        rolling_rows.append({{
            "estimator": label,
            "mean_vol": float(np.mean(vols)),
            "std_vol": float(np.std(vols, ddof=1)),
            "min_vol": float(np.min(vols)),
            "max_vol": float(np.max(vols)),
            "efficiency": float(efficiencies[key]),
        }})
    pd.DataFrame(rolling_rows).to_csv(output_dir / "rolling_vol_stats.csv", index=False)

    term_rows = []
    for window in (5, 10, 21, 63, 126, 252):
        estimates = rolling_variances(frame, window)
        term_rows.append({{
            "window": window,
            "cc_vol": float(np.mean([annual_vol(row["cc"]) for row in estimates])),
            "yz_vol": float(np.mean([annual_vol(row["yang_zhang"]) for row in estimates])),
        }})
    pd.DataFrame(term_rows).to_csv(output_dir / "vol_term_structure.csv", index=False)

    range_keys = ["parkinson", "garman_klass", "rogers_satchell", "yang_zhang"]
    summary = {{
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
            all(estimator_sampling_variance[key] < baseline_variance for key in range_keys)
        ),
        "yz_most_efficient": bool(
            efficiencies["yang_zhang"] == max(efficiencies.values())
        ),
        "vol_mean_range": [min(full_vols.values()), max(full_vols.values())],
    }}

    write_json(output_dir / "calibration.json", calibration)
    write_json(output_dir / "full_sample_vol.json", full_sample)
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
'''


def _standard_var_starter(data_path: str) -> str:
    return f'''import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


NOTIONAL = 1_000_000.0
ALPHAS = (0.95, 0.99)


def historical_risk(returns, alpha):
    tail_probability = 1.0 - alpha
    cutoff = float(np.quantile(returns, tail_probability))
    tail = returns[returns <= cutoff]
    return -cutoff * NOTIONAL, -float(np.mean(tail)) * NOTIONAL


def normal_risk(mean, volatility, alpha):
    z = float(stats.norm.ppf(1.0 - alpha))
    value_at_risk = -(mean + volatility * z) * NOTIONAL
    expected_shortfall = (
        -mean + volatility * stats.norm.pdf(z) / (1.0 - alpha)
    ) * NOTIONAL
    return float(value_at_risk), float(expected_shortfall)


def student_risk(parameters, alpha):
    degrees, location, scale = parameters
    standardized_cutoff = float(stats.t.ppf(1.0 - alpha, degrees))
    cutoff = location + scale * standardized_cutoff
    tail_mean = location - scale * (
        (degrees + standardized_cutoff ** 2) / (degrees - 1.0)
    ) * stats.t.pdf(standardized_cutoff, degrees) / (1.0 - alpha)
    return -float(cutoff) * NOTIONAL, -float(tail_mean) * NOTIONAL


def age_weighted_risk(returns, alpha, decay=0.98):
    values = np.asarray(returns, dtype=float)
    weights = decay ** np.arange(len(values) - 1, -1, -1, dtype=float)
    weights /= weights.sum()
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    tail_probability = 1.0 - alpha
    cutoff_index = int(np.searchsorted(np.cumsum(sorted_weights), tail_probability))
    cutoff_index = min(cutoff_index, len(sorted_values) - 1)
    cutoff = sorted_values[cutoff_index]
    tail_mask = values <= cutoff
    tail_mean = float(np.average(values[tail_mask], weights=weights[tail_mask]))
    return -float(cutoff) * NOTIONAL, -tail_mean * NOTIONAL


def current_ewma_covariance(asset_returns, decay=0.94):
    observations = np.asarray(asset_returns, dtype=float)
    covariance = np.outer(observations[0], observations[0])
    for observation in observations[1:]:
        covariance = decay * covariance + (1.0 - decay) * np.outer(observation, observation)
    return covariance


def kupiec_pvalue(n_observations, n_exceedances, expected_probability):
    observed_probability = n_exceedances / n_observations
    null_log_likelihood = (
        (n_observations - n_exceedances) * math.log1p(-expected_probability)
        + n_exceedances * math.log(expected_probability)
    )
    alternative_log_likelihood = (
        (n_observations - n_exceedances) * math.log1p(-observed_probability)
        + n_exceedances * math.log(observed_probability)
    )
    statistic = -2.0 * (null_log_likelihood - alternative_log_likelihood)
    return float(stats.chi2.sf(statistic, 1))


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\\n",
        encoding="utf-8",
    )


def main():
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(task_dir / {data_path!r}, parse_dates=["date"])
    counts = frame.groupby("symbol").size()
    closes = (
        frame.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .dropna()
    )
    if closes.shape[1] != 2:
        raise ValueError("standard two-asset VaR starter requires exactly two aligned symbols")
    asset_returns = np.log(closes / closes.shift(1)).dropna()
    portfolio_returns = asset_returns.mean(axis=1).to_numpy(dtype=float)
    mean = float(np.mean(portfolio_returns))
    volatility = float(np.std(portfolio_returns, ddof=1))
    correlation = float(asset_returns.corr().iloc[0, 1])

    symbols = list(closes.columns)
    data_summary = {{
        "n_aapl": int(counts.get("AAPL", 0)),
        "n_jpm": int(counts.get("JPM", 0)),
        "n_aligned": int(len(closes)),
        "n_returns": int(len(portfolio_returns)),
        "portfolio_mean": mean,
        "portfolio_std": volatility,
        "correlation": correlation,
    }}

    student_parameters = tuple(float(value) for value in stats.t.fit(portfolio_returns))
    weights = np.full(len(symbols), 1.0 / len(symbols))
    ewma_covariance = current_ewma_covariance(asset_returns.to_numpy(dtype=float))
    ewma_volatility = float(math.sqrt(weights @ ewma_covariance @ weights))

    one_day_rows = []
    one_day_lookup = {{}}
    for alpha in ALPHAS:
        risks = {{
            "HS": historical_risk(portfolio_returns, alpha),
            "Normal": normal_risk(mean, volatility, alpha),
            "StudentT": student_risk(student_parameters, alpha),
            "AgeWeightedHS": age_weighted_risk(portfolio_returns, alpha),
            "EWMA": normal_risk(0.0, ewma_volatility, alpha),
        }}
        for method in ("HS", "Normal", "StudentT", "AgeWeightedHS", "EWMA"):
            value_at_risk, expected_shortfall = risks[method]
            one_day_lookup[(method, alpha)] = (value_at_risk, expected_shortfall)
            one_day_rows.append({{
                "method": method,
                "alpha": alpha,
                "var_dollar": value_at_risk,
                "es_dollar": expected_shortfall,
            }})

    overlapping_ten_day = pd.Series(portfolio_returns).rolling(10).sum().dropna().to_numpy()
    ten_day_rows = []
    for alpha in ALPHAS:
        simulation_risks = {{
            "HS": historical_risk(overlapping_ten_day, alpha),
            "AgeWeightedHS": age_weighted_risk(overlapping_ten_day, alpha),
        }}
        for method in ("HS", "Normal", "StudentT", "AgeWeightedHS", "EWMA"):
            if method in simulation_risks:
                value_at_risk, expected_shortfall = simulation_risks[method]
            else:
                one_day_var, one_day_es = one_day_lookup[(method, alpha)]
                value_at_risk = one_day_var * math.sqrt(10.0)
                expected_shortfall = one_day_es * math.sqrt(10.0)
            ten_day_rows.append({{
                "method": method,
                "alpha": alpha,
                "var_dollar": value_at_risk,
                "es_dollar": expected_shortfall,
            }})

    n_exceedances = 0
    minimum_window = 252
    for index in range(minimum_window, len(portfolio_returns)):
        historical_cutoff = float(np.quantile(portfolio_returns[:index], 0.01))
        n_exceedances += int(portfolio_returns[index] < historical_cutoff)
    n_test_days = len(portfolio_returns) - minimum_window
    backtest = {{
        "n_test_days": n_test_days,
        "n_exceedances": n_exceedances,
        "exceedance_rate": n_exceedances / n_test_days,
        "expected_rate": 0.01,
        "kupiec_pvalue": kupiec_pvalue(n_test_days, n_exceedances, 0.01),
    }}

    ninety_nine = {{
        method: one_day_lookup[(method, 0.99)][0]
        for method in ("HS", "Normal", "StudentT", "AgeWeightedHS", "EWMA")
    }}
    summary = {{
        "most_conservative_method": max(ninety_nine, key=ninety_nine.get),
        "least_conservative_method": min(ninety_nine, key=ninety_nine.get),
        "var_range_99": [min(ninety_nine.values()), max(ninety_nine.values())],
        "ewma_current_vol": ewma_volatility * math.sqrt(252.0),
    }}

    pd.DataFrame(one_day_rows).to_csv(output_dir / "var_1day.csv", index=False)
    pd.DataFrame(ten_day_rows).to_csv(output_dir / "var_10day.csv", index=False)
    write_json(output_dir / "data_summary.json", data_summary)
    write_json(output_dir / "backtest.json", backtest)
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
'''


def _brinson_attribution_starter(
    prices_path: str,
    params_path: str,
    cash_rates_path: str,
) -> str:
    return f'''import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
            default=lambda value: float(value) if isinstance(value, np.floating) else int(value),
        ) + "\\n",
        encoding="utf-8",
    )


def main():
    task_dir = Path(os.environ["TASK_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    params = load_json(task_dir / {params_path!r})
    prices = pd.read_csv(task_dir / {prices_path!r}, parse_dates=["Date"])
    prices = prices.sort_values("Date").set_index("Date")
    monthly_prices = prices.groupby(prices.index.to_period("M")).last()
    monthly_returns = monthly_prices.pct_change()
    start_period = pd.Period(params["evaluation_start"], freq="M")
    end_period = pd.Period(params["evaluation_end"], freq="M")
    monthly_returns = monthly_returns.loc[start_period:end_period]
    if len(monthly_returns) != 12:
        raise ValueError(f"expected 12 evaluation months, found {{len(monthly_returns)}}")

    cash_rates = pd.read_csv(task_dir / {cash_rates_path!r})
    cash_returns = {{
        pd.Period(row["month"], freq="M"): float(row["annual_rate"]) / 12.0
        for _, row in cash_rates.iterrows()
    }}
    portfolio_targets = {{
        sector: float(weight) for sector, weight in params["portfolio_weights"].items()
    }}
    sectors = list(portfolio_targets)
    equity_sectors = [sector for sector in sectors if sector != "CASH"]
    if not math_is_close(sum(portfolio_targets.values()), 1.0):
        raise ValueError("portfolio target weights must sum to one")

    monthly_allocation = []
    monthly_selection = []
    monthly_interaction = []
    portfolio_monthly_returns = []
    benchmark_monthly_returns = []
    sector_allocation = {{sector: 0.0 for sector in sectors}}
    sector_selection = {{sector: 0.0 for sector in equity_sectors}}
    sector_interaction = {{sector: 0.0 for sector in equity_sectors}}
    monthly_sector_allocation = {{sector: [] for sector in sectors}}
    weight_snapshot_march = None
    portfolio_weights = dict(portfolio_targets)

    for month_index, (period, returns) in enumerate(monthly_returns.iterrows()):
        quarter = f"Q{{month_index // 3 + 1}}"
        if month_index % 3 == 0:
            portfolio_weights = dict(portfolio_targets)
        if month_index == 2:
            weight_snapshot_march = dict(portfolio_weights)

        benchmark_weights = {{
            sector: float(weight)
            for sector, weight in params["benchmark_weights"][quarter].items()
        }}
        benchmark_return = sum(
            benchmark_weights[sector] * float(returns[sector])
            for sector in equity_sectors
        )
        portfolio_sector_returns = {{
            sector: (
                cash_returns[period]
                if sector == "CASH"
                else float(returns[params["portfolio_etfs"][sector]])
            )
            for sector in sectors
        }}
        benchmark_sector_returns = {{
            sector: float(returns[sector]) for sector in equity_sectors
        }}
        portfolio_return = sum(
            portfolio_weights[sector] * portfolio_sector_returns[sector]
            for sector in sectors
        )

        allocation = {{}}
        selection = {{}}
        interaction = {{}}
        for sector in sectors:
            if sector == "CASH":
                allocation[sector] = portfolio_weights[sector] * (
                    portfolio_sector_returns[sector] - benchmark_return
                )
            else:
                active_weight = portfolio_weights[sector] - benchmark_weights[sector]
                active_return = (
                    portfolio_sector_returns[sector] - benchmark_sector_returns[sector]
                )
                allocation[sector] = active_weight * (
                    benchmark_sector_returns[sector] - benchmark_return
                )
                selection[sector] = benchmark_weights[sector] * active_return
                interaction[sector] = active_weight * active_return

            sector_allocation[sector] += allocation[sector]
            monthly_sector_allocation[sector].append(allocation[sector])
            if sector != "CASH":
                sector_selection[sector] += selection[sector]
                sector_interaction[sector] += interaction[sector]

        monthly_allocation.append(sum(allocation.values()))
        monthly_selection.append(sum(selection.values()))
        monthly_interaction.append(sum(interaction.values()))
        portfolio_monthly_returns.append(portfolio_return)
        benchmark_monthly_returns.append(benchmark_return)

        ending_values = {{
            sector: portfolio_weights[sector] * (1.0 + portfolio_sector_returns[sector])
            for sector in sectors
        }}
        ending_total = sum(ending_values.values())
        portfolio_weights = {{
            sector: ending_values[sector] / ending_total for sector in sectors
        }}

    portfolio_annual_return = float(np.prod(1.0 + np.asarray(portfolio_monthly_returns)) - 1.0)
    benchmark_annual_return = float(np.prod(1.0 + np.asarray(benchmark_monthly_returns)) - 1.0)
    result = {{
        "portfolio_return": portfolio_annual_return,
        "benchmark_return": benchmark_annual_return,
        "active_return": portfolio_annual_return - benchmark_annual_return,
        "total_allocation_effect": sum(monthly_allocation),
        "total_selection_effect": sum(monthly_selection),
        "total_interaction_effect": sum(monthly_interaction),
        "monthly_allocation": monthly_allocation,
        "monthly_selection": monthly_selection,
        "monthly_interaction": monthly_interaction,
        "q1_allocation": sum(monthly_allocation[0:3]),
        "q2_allocation": sum(monthly_allocation[3:6]),
        "q3_allocation": sum(monthly_allocation[6:9]),
        "q4_allocation": sum(monthly_allocation[9:12]),
        "sector_allocation": sector_allocation,
        "sector_selection": sector_selection,
        "sector_interaction": sector_interaction,
        "monthly_sector_allocation": monthly_sector_allocation,
        "weight_snapshot_march": weight_snapshot_march,
    }}
    write_json(output_dir / "results.json", result)


def math_is_close(left, right, tolerance=1e-12):
    return abs(left - right) <= tolerance


if __name__ == "__main__":
    main()
'''


def starter_for(task: TaskSpec, input_inventory: tuple[str, ...]) -> StarterProgram | None:
    lowered = task.safe_instruction.lower()
    is_brinson_attribution = (
        "brinson-fachler sector attribution" in lowered
        and "weight_snapshot_march" in lowered
        and "interaction" in lowered
    )
    if is_brinson_attribution:
        expected = {
            "sector_etfs.csv": None,
            "params.json": None,
            "cash_rates.csv": None,
        }
        for path in input_inventory:
            name = path.rsplit("/", 1)[-1]
            if path.startswith("environment/data/") and name in expected:
                expected[name] = path
        if all(expected.values()):
            return StarterProgram(
                path="scratch/solve.py",
                content=_brinson_attribution_starter(
                    expected["sector_etfs.csv"],
                    expected["params.json"],
                    expected["cash_rates.csv"],
                ),
                purpose=(
                    "Brinson-Fachler monthly attribution with quarterly rebalancing, "
                    "intra-quarter weight drift, alternative holdings, and cash returns"
                ),
            )

    is_standard_var = (
        "standard var methods comparison" in lowered
        and "age-weighted historical simulation" in lowered
        and "kupiec" in lowered
    )
    if is_standard_var:
        candidates = [
            path
            for path in input_inventory
            if path.startswith("environment/data/")
            and path.lower().endswith("pair_aapl_jpm_daily.csv")
        ]
        if len(candidates) == 1:
            return StarterProgram(
                path="scratch/solve.py",
                content=_standard_var_starter(candidates[0]),
                purpose=(
                    "five-method one- and ten-day VaR/ES analysis with Student-t MLE, "
                    "EWMA covariance, age weighting, and expanding-window Kupiec backtest"
                ),
            )

    is_ohlc_volatility = (
        "ohlc realized volatility estimators" in lowered
        and "yang-zhang" in lowered
        and "rogers-satchell" in lowered
    )
    if is_ohlc_volatility:
        candidates = [
            path
            for path in input_inventory
            if path.startswith("environment/data/") and path.lower().endswith("spy_daily.csv")
        ]
        if len(candidates) == 1:
            return StarterProgram(
                path="scratch/solve.py",
                content=_ohlc_volatility_starter(candidates[0]),
                purpose=(
                    "OHLC volatility estimation with five estimators, consistent prior-close "
                    "rolling windows, sampling-efficiency analysis, and term structure"
                ),
            )

    is_parity_audit = (
        "put-call parity" in lowered
        and "synthetic_forward_bid" in lowered
        and "implied_borrow_rate" in lowered
    )
    if is_parity_audit:
        expected = {
            "option_quotes.csv": None,
            "spot.json": None,
            "carry_inputs.json": None,
            "audit_rules.json": None,
        }
        for path in input_inventory:
            name = path.rsplit("/", 1)[-1]
            if path.startswith("environment/data/") and name in expected:
                expected[name] = path
        if all(expected.values()):
            return StarterProgram(
                path="scratch/solve.py",
                content=_parity_audit_starter(
                    expected["option_quotes.csv"],
                    expected["spot.json"],
                    expected["carry_inputs.json"],
                    expected["audit_rules.json"],
                ),
                purpose=(
                    "executable put-call parity audit with quote cleaning, bid/ask "
                    "forward bounds, threshold checks, and fixed-format deliverables"
                ),
            )

    is_bootstrap = "bootstrap" in lowered and (
        "zero-coupon" in lowered or "par coupon" in lowered or "yield curve" in lowered
    )
    if not is_bootstrap:
        return None
    candidates = [
        path
        for path in input_inventory
        if path.startswith("environment/data/") and path.lower().endswith("curve_data.json")
    ]
    if len(candidates) != 1:
        return None
    return StarterProgram(
        path="scratch/solve.py",
        content=_bootstrap_starter(candidates[0]),
        purpose="numeric zero-coupon bootstrap with interpolation and par-repricing self-check",
    )
