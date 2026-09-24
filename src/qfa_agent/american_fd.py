"""Crank--Nicolson/PSOR workflow for American-option research tasks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Sequence


def _price_fd_option(
    *,
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    stock_steps: int,
    time_steps: int,
    option_type: str,
    exercise_type: str,
    dividend_times: Sequence[float],
    dividend_amounts: Sequence[float],
    omega: float,
    tolerance: float,
    max_iterations: int,
    capture_grid: bool = False,
    capture_boundary: bool = False,
) -> dict[str, object]:
    import numpy as np

    if option_type not in {"put", "call"} or exercise_type not in {"american", "european"}:
        raise ValueError("unsupported option or exercise type")
    if len(dividend_times) != len(dividend_amounts):
        raise ValueError("dividend times and amounts must have equal length")
    maximum_stock = 3.0 * strike
    delta_stock = maximum_stock / stock_steps
    delta_time = maturity / time_steps
    stock_grid = np.linspace(0.0, maximum_stock, stock_steps + 1)
    payoff = (
        np.maximum(strike - stock_grid, 0.0)
        if option_type == "put" else np.maximum(stock_grid - strike, 0.0)
    )
    values = payoff.copy()

    boundary = np.zeros(time_steps + 1, dtype=float)
    if capture_boundary and option_type == "put":
        boundary[-1] = strike
    sample_stock_indexes = list(range(0, stock_steps + 1, 5))
    if stock_steps not in sample_stock_indexes:
        sample_stock_indexes.append(stock_steps)
    sample_time_indexes = list(range(0, time_steps + 1, 10))
    if time_steps not in sample_time_indexes:
        sample_time_indexes.append(time_steps)
    sampled_grid = np.zeros((len(sample_stock_indexes), len(sample_time_indexes)))
    if capture_grid:
        terminal_column = sample_time_indexes.index(time_steps)
        sampled_grid[:, terminal_column] = values[sample_stock_indexes]

    nodes = np.arange(1, stock_steps, dtype=float)
    alpha = 0.25 * delta_time * (volatility * volatility * nodes * nodes - rate * nodes)
    beta = -0.5 * delta_time * (volatility * volatility * nodes * nodes + rate)
    gamma = 0.25 * delta_time * (volatility * volatility * nodes * nodes + rate * nodes)
    lower = -alpha
    diagonal = 1.0 - beta
    upper = -gamma
    explicit_diagonal = 1.0 + beta
    dividend_steps = [int(round(time / delta_time)) for time in dividend_times]

    for time_index in range(time_steps - 1, -1, -1):
        tau_new = maturity - time_index * delta_time
        tau_old = maturity - (time_index + 1) * delta_time
        for dividend_index, dividend_step in enumerate(dividend_steps):
            if time_index + 1 != dividend_step:
                continue
            amount = float(dividend_amounts[dividend_index])
            adjusted = np.empty(stock_steps + 1, dtype=float)
            for node in range(stock_steps + 1):
                shifted = stock_grid[node] - amount
                if shifted <= 0.0:
                    adjusted[node] = strike * math.exp(-rate * tau_old) if option_type == "put" else 0.0
                elif shifted >= maximum_stock:
                    adjusted[node] = 0.0 if option_type == "put" else maximum_stock - strike * math.exp(-rate * tau_old)
                else:
                    left = min(int(shifted / delta_stock), stock_steps - 1)
                    fraction = (shifted - stock_grid[left]) / delta_stock
                    adjusted[node] = values[left] + fraction * (values[left + 1] - values[left])
            values = adjusted
            break

        if option_type == "put":
            new_left, new_right = strike * math.exp(-rate * tau_new), 0.0
            old_left, old_right = strike * math.exp(-rate * tau_old), 0.0
        else:
            new_left, new_right = 0.0, maximum_stock - strike * math.exp(-rate * tau_new)
            old_left, old_right = 0.0, maximum_stock - strike * math.exp(-rate * tau_old)
        values[0], values[-1] = old_left, old_right
        right_hand = (
            alpha * values[:-2]
            + explicit_diagonal * values[1:-1]
            + gamma * values[2:]
        )
        right_hand[0] += alpha[0] * new_left
        right_hand[-1] += gamma[-1] * new_right

        interior = values[1:-1].copy()
        for _ in range(max_iterations):
            maximum_change = 0.0
            for index in range(stock_steps - 1):
                candidate_sum = right_hand[index]
                if index > 0:
                    candidate_sum -= lower[index] * interior[index - 1]
                if index < stock_steps - 2:
                    candidate_sum -= upper[index] * interior[index + 1]
                solved = candidate_sum / diagonal[index]
                candidate = interior[index] + omega * (solved - interior[index])
                if exercise_type == "american":
                    candidate = max(candidate, float(payoff[index + 1]))
                maximum_change = max(maximum_change, abs(candidate - interior[index]))
                interior[index] = candidate
            if maximum_change < tolerance:
                break
        else:
            raise ValueError("PSOR did not converge within max_iterations")

        values[1:-1] = interior
        values[0], values[-1] = new_left, new_right
        if capture_boundary and option_type == "put" and exercise_type == "american":
            for node in range(stock_steps, 0, -1):
                if payoff[node] > 0.0 and abs(values[node] - payoff[node]) < 1.0e-6:
                    boundary[time_index] = stock_grid[node]
                    break
        if capture_grid and time_index in sample_time_indexes:
            sampled_grid[:, sample_time_indexes.index(time_index)] = values[sample_stock_indexes]

    center = int(round(spot / delta_stock))
    result: dict[str, object] = {
        "value": float(values[center]),
        "delta": float((values[center + 1] - values[center - 1]) / (2.0 * delta_stock)),
    }
    if capture_grid:
        result.update(
            grid=sampled_grid,
            sample_stock=[float(stock_grid[index]) for index in sample_stock_indexes],
            sample_time=[float(index * delta_time) for index in sample_time_indexes],
        )
    if capture_boundary:
        result.update(
            boundary=boundary,
            boundary_time=np.arange(time_steps + 1, dtype=float) * delta_time,
        )
    return result


def write_american_option_fd_outputs(
    output_dir: str | Path,
    *,
    spot: float = 100.0,
    strike: float = 100.0,
    rate: float = 0.05,
    volatility: float = 0.30,
    maturity: float = 1.0,
    dividend_times: Sequence[float] = (0.25, 0.75),
    dividend_amounts: Sequence[float] = (2.50, 2.50),
    fine_stock_steps: int = 300,
    fine_time_steps: int = 600,
    coarse_stock_steps: int = 150,
    coarse_time_steps: int = 300,
    omega: float = 1.2,
    tolerance: float = 1.0e-8,
    max_iterations: int = 10_000,
) -> dict[str, object]:
    """Price the full American/European discrete-dividend comparison bundle."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    common = dict(
        spot=float(spot), strike=float(strike), rate=float(rate),
        volatility=float(volatility), maturity=float(maturity),
        stock_steps=int(fine_stock_steps), time_steps=int(fine_time_steps),
        omega=float(omega), tolerance=float(tolerance),
        max_iterations=int(max_iterations),
    )
    with_dividends: dict[str, dict[str, object]] = {}
    for option_type in ("put", "call"):
        for exercise_type in ("american", "european"):
            name = f"{exercise_type}_{option_type}"
            with_dividends[name] = _price_fd_option(
                **common, option_type=option_type, exercise_type=exercise_type,
                dividend_times=dividend_times, dividend_amounts=dividend_amounts,
                capture_grid=name == "american_put",
                capture_boundary=name == "american_put",
            )
    without_dividends: dict[str, dict[str, object]] = {}
    for option_type in ("put", "call"):
        for exercise_type in ("american", "european"):
            name = f"{exercise_type}_{option_type}"
            without_dividends[name] = _price_fd_option(
                **common, option_type=option_type, exercise_type=exercise_type,
                dividend_times=(), dividend_amounts=(),
            )
    coarse = _price_fd_option(
        spot=float(spot), strike=float(strike), rate=float(rate),
        volatility=float(volatility), maturity=float(maturity),
        stock_steps=int(coarse_stock_steps), time_steps=int(coarse_time_steps),
        option_type="put", exercise_type="american",
        dividend_times=dividend_times, dividend_amounts=dividend_amounts,
        omega=float(omega), tolerance=float(tolerance), max_iterations=int(max_iterations),
    )

    fine_value = float(with_dividends["american_put"]["value"])
    coarse_value = float(coarse["value"])
    richardson = (4.0 * fine_value - coarse_value) / 3.0
    fine_error = abs(fine_value - richardson)
    convergence_ratio = (
        abs(coarse_value - richardson) / fine_error if fine_error > 1.0e-10 else math.inf
    )
    option_values = {
        name: float(result["value"]) for name, result in with_dividends.items()
    }
    option_values.update({
        f"{name}_no_div": float(result["value"])
        for name, result in without_dividends.items()
    })
    no_dividend_call_difference = abs(
        option_values["american_call_no_div"] - option_values["european_call_no_div"]
    )
    boundary = np.asarray(with_dividends["american_put"]["boundary"], dtype=float)
    boundary_time = np.asarray(
        with_dividends["american_put"]["boundary_time"], dtype=float
    )
    summary = {
        "american_geq_european_put": bool(
            option_values["american_put"] >= option_values["european_put"]
        ),
        "american_geq_european_call": bool(
            option_values["american_call"] >= option_values["european_call"]
        ),
        "no_div_call_diff": float(no_dividend_call_difference),
        "early_exercise_premium_put": float(
            option_values["american_put"] - option_values["european_put"]
        ),
        "early_exercise_premium_call": float(
            option_values["american_call"] - option_values["european_call"]
        ),
        "exercise_boundary_at_T": float(boundary[-1]),
        "exercise_boundary_at_0": float(boundary[0]),
        "richardson_estimate": float(richardson),
        "convergence_ratio": float(convergence_ratio),
    }

    # Validate structure, finance inequalities, and an alternate-resolution path.
    if not all(math.isfinite(value) and value > 0.0 for value in option_values.values()):
        raise AssertionError("option values must be finite and positive")
    if not summary["american_geq_european_put"] or not summary["american_geq_european_call"]:
        raise AssertionError("American values violate the European lower bound")
    if no_dividend_call_difference >= 1.0e-3:
        raise AssertionError("no-dividend American call does not reconcile to European call")
    if option_values["european_put"] <= option_values["european_put_no_div"]:
        raise AssertionError("cash dividends should increase a European put")
    if option_values["european_call"] >= option_values["european_call_no_div"]:
        raise AssertionError("cash dividends should decrease a European call")
    if not (2.0 < convergence_ratio < 8.0):
        raise AssertionError("coarse/fine grids do not show second-order convergence")
    if len(boundary) != fine_time_steps + 1 or np.any(boundary < 0.0):
        raise AssertionError("exercise boundary has invalid structure")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "option_values.json").write_text(
        json.dumps(option_values, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    grid = np.asarray(with_dividends["american_put"]["grid"], dtype=float)
    sample_stock = with_dividends["american_put"]["sample_stock"]
    sample_time = with_dividends["american_put"]["sample_time"]
    with (target / "american_put_grid.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["S", *[f"{value:.4f}" for value in sample_time]])
        for stock_index, stock_value in enumerate(sample_stock):
            writer.writerow([
                f"{stock_value:.2f}",
                *[f"{grid[stock_index, time_index]:.6f}" for time_index in range(len(sample_time))],
            ])
    with (target / "early_exercise_boundary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["t", "S_star"])
        for time_value, stock_value in zip(boundary_time, boundary):
            writer.writerow([f"{time_value:.6f}", f"{stock_value:.6f}"])
    with (target / "greeks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["option_type", "delta"])
        for name in ("american_put", "american_call", "european_put", "european_call"):
            writer.writerow([name, f"{float(with_dividends[name]['delta']):.6f}"])
    with (target / "convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream); writer.writerow(["grid", "price"])
        writer.writerow(["fine", f"{fine_value:.6f}"])
        writer.writerow(["coarse", f"{coarse_value:.6f}"])
        writer.writerow(["richardson", f"{richardson:.6f}"])
    figure, axis = plt.subplots(figsize=(10, 6))
    valid = boundary > 0.0
    axis.plot(boundary_time[valid], boundary[valid], linewidth=1.5)
    axis.axhline(strike, color="red", linestyle="--", alpha=0.5, label=f"K={strike:g}")
    axis.set_xlabel("Time t"); axis.set_ylabel("Critical Stock Price S*(t)")
    axis.set_title("Early Exercise Boundary (American Put)"); axis.legend(); axis.grid(alpha=0.3)
    figure.tight_layout(); figure.savefig(target / "exercise_boundary.png", dpi=150); plt.close(figure)
    (target / "summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "option_values": option_values, "summary": summary,
        "greeks": {name: float(with_dividends[name]["delta"]) for name in with_dividends},
    }
