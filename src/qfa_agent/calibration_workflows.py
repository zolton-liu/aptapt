"""Validated calibration and factor-research workflow operators.

The operators in this module are data-driven and perform their own structural,
numerical, and independent consistency checks before publishing artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


def write_mean_reverting_jump_diffusion_outputs(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    rate_column: str = "DGS10",
    periods_per_year: int = 252,
    jump_threshold: float = 3.0,
    horizons: Sequence[float] = (0.25, 0.50, 1.00, 2.00, 5.00),
    paths: int = 10_000,
    seed: int = 42,
) -> dict[str, object]:
    """Calibrate a log-OU jump diffusion and write its standard artifact bundle."""

    import numpy as np
    import pandas as pd

    frame = pd.read_csv(data_path)
    if rate_column not in frame:
        raise ValueError(f"missing rate column: {rate_column}")
    rates = pd.to_numeric(frame[rate_column], errors="coerce").dropna().to_numpy(float)
    rates = rates[rates > 0.0] / 100.0
    if rates.size < 3 or not np.isfinite(rates).all():
        raise ValueError("calibration needs at least three finite positive rates")

    log_rates = np.log(rates)
    x_left, x_right = log_rates[:-1], log_rates[1:]
    dt = 1.0 / float(periods_per_year)
    centered_left = x_left - x_left.mean()
    denominator = float(centered_left @ centered_left)
    if denominator <= 0.0:
        raise ValueError("rate history has no usable log-level variation")
    slope = float(centered_left @ (x_right - x_right.mean()) / denominator)
    intercept = float(x_right.mean() - slope * x_left.mean())
    if not 0.0 < slope < 1.0:
        raise ValueError("OLS slope does not imply a positive mean-reversion speed")

    kappa = -math.log(slope) / dt
    theta = intercept / (1.0 - slope)
    residuals = x_right - (intercept + slope * x_left)
    residual_variance = float(np.var(residuals, ddof=1))
    sigma = math.sqrt(
        residual_variance * 2.0 * kappa / (1.0 - math.exp(-2.0 * kappa * dt))
    )
    residual_std = float(np.std(residuals, ddof=1))
    if min(kappa, sigma, residual_std) <= 0.0:
        raise ValueError("OU calibration did not produce positive scale parameters")

    jump_mask = np.abs(residuals / residual_std) > float(jump_threshold)
    jump_residuals = residuals[jump_mask]
    n_jumps = int(jump_mask.sum())
    annual_intensity = n_jumps / (len(residuals) * dt)
    jump_mean = float(jump_residuals.mean()) if n_jumps else 0.0
    jump_std = float(jump_residuals.std(ddof=1)) if n_jumps > 1 else 0.0
    x0 = float(log_rates[-1])
    stationary_mean = theta + annual_intensity * jump_mean / kappa
    stationary_variance = (
        sigma * sigma + annual_intensity * (jump_mean * jump_mean + jump_std * jump_std)
    ) / (2.0 * kappa)

    def conditional_mean(tau: float, *, with_jumps: bool = True) -> float:
        decay = math.exp(-kappa * tau)
        jump_drift = annual_intensity * jump_mean if with_jumps else 0.0
        return theta + (x0 - theta) * decay + jump_drift * (1.0 - decay) / kappa

    def conditional_variance(tau: float, *, with_jumps: bool = True) -> float:
        innovation = sigma * sigma
        if with_jumps:
            innovation += annual_intensity * (jump_mean * jump_mean + jump_std * jump_std)
        return innovation * (1.0 - math.exp(-2.0 * kappa * tau)) / (2.0 * kappa)

    moment_rows: list[dict[str, float]] = []
    for raw_tau in horizons:
        tau = float(raw_tau)
        if tau <= 0.0:
            raise ValueError("all horizons must be positive")
        mean_x = conditional_mean(tau)
        variance_x = conditional_variance(tau)
        mean_s = math.exp(mean_x + 0.5 * variance_x)
        variance_s = mean_s * mean_s * math.expm1(variance_x)
        moment_rows.append(
            {"tau": tau, "E_X": mean_x, "Var_X": variance_x,
             "E_S": mean_s, "Var_S": variance_s}
        )

    rng = np.random.RandomState(int(seed))
    for row in moment_rows:
        steps = int(round(row["tau"] / dt))
        simulated = np.full(int(paths), x0, dtype=float)
        for _ in range(steps):
            diffusion = rng.randn(int(paths))
            counts = rng.poisson(annual_intensity * dt, int(paths))
            jump_sum = np.zeros(int(paths), dtype=float)
            for index in np.flatnonzero(counts):
                count = int(counts[index])
                jump_sum[index] = float(
                    rng.normal(jump_mean, max(jump_std, 1.0e-10), count).sum()
                )
            simulated += (
                kappa * (theta - simulated) * dt
                + sigma * math.sqrt(dt) * diffusion
                + jump_sum
            )
        row["E_X_mc"] = float(simulated.mean())
        row["Var_X_mc"] = float(simulated.var(ddof=1))

    moment_frame = pd.DataFrame(moment_rows)
    forward_rows = []
    for tau in (float(value) for value in horizons):
        forward_rows.append(
            {
                "tau": tau,
                "forward_rate_jd": math.exp(
                    conditional_mean(tau) + 0.5 * conditional_variance(tau)
                ),
                "forward_rate_ou": math.exp(
                    conditional_mean(tau, with_jumps=False)
                    + 0.5 * conditional_variance(tau, with_jumps=False)
                ),
            }
        )
    forward_frame = pd.DataFrame(forward_rows)

    mean_errors = np.abs(moment_frame["E_X"] - moment_frame["E_X_mc"]) / np.abs(
        moment_frame["E_X"]
    )
    variance_errors = np.abs(
        moment_frame["Var_X"] - moment_frame["Var_X_mc"]
    ) / moment_frame["Var_X"]
    one_year = forward_frame.loc[np.isclose(forward_frame["tau"], 1.0)]
    jump_impact = 0.0
    if not one_year.empty:
        jump_impact = float(
            abs(one_year["forward_rate_jd"].iloc[0] - one_year["forward_rate_ou"].iloc[0])
            / one_year["forward_rate_ou"].iloc[0]
        )

    calibration = {
        "n_obs": int(rates.size),
        "n_changes": int(residuals.size),
        "rate_mean": float(rates.mean()),
        "rate_std": float(rates.std(ddof=1)),
        "log_rate_mean": float(log_rates.mean()),
        "log_rate_std": float(log_rates.std(ddof=1)),
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "lambda": float(annual_intensity),
        "mu_J": float(jump_mean),
        "sigma_J": float(jump_std),
        "n_jumps": n_jumps,
        "x0": x0,
        "S0": float(rates[-1]),
        "stationary_mean_X": float(stationary_mean),
        "stationary_var_X": float(stationary_variance),
    }
    half_life = math.log(2.0) / kappa
    summary = {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "lambda_annual": float(annual_intensity),
        "half_life_years": float(half_life),
        "stationary_vol_X": math.sqrt(stationary_variance),
        "feller_ratio": float(2.0 * kappa * theta / (sigma * sigma)),
        "max_mc_mean_relerr": float(mean_errors.max()),
        "max_mc_var_relerr": float(variance_errors.max()),
        "jump_impact_on_forward": jump_impact,
        "mean_reversion_confirmed": bool(kappa > 0.0 and half_life < 20.0),
        "jumps_detected": bool(n_jumps > 0),
    }

    # Structure, numerical, and independent cross-checks before publication.
    if calibration["n_changes"] != calibration["n_obs"] - 1:
        raise AssertionError("change count is inconsistent with observations")
    if not np.isfinite(moment_frame.to_numpy(float)).all():
        raise AssertionError("conditional moments contain non-finite values")
    if np.any(np.diff(moment_frame["Var_X"].to_numpy()) < -1.0e-10):
        raise AssertionError("conditional variance is not monotone in horizon")
    if not np.allclose(np.exp(x0), calibration["S0"], rtol=1.0e-12):
        raise AssertionError("log-level and final rate do not reconcile")
    if summary["max_mc_mean_relerr"] >= 0.05 or summary["max_mc_var_relerr"] >= 0.15:
        raise AssertionError("Monte Carlo does not independently confirm analytic moments")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "calibration.json", calibration)
    moment_frame.to_csv(target / "conditional_moments.csv", index=False)
    forward_frame.to_csv(target / "forward_curve.csv", index=False)
    _write_json(target / "summary.json", summary)
    return {
        "calibration": calibration,
        "conditional_moments": moment_frame,
        "forward_curve": forward_frame,
        "summary": summary,
    }


DEFAULT_EQUITY_SECTORS = {
    "AAPL": "IT", "MSFT": "IT", "GOOGL": "IT",
    "AMZN": "Services", "MCD": "Services", "UNH": "Services",
    "JPM": "Finance", "GS": "Finance", "V": "Finance", "BRK-B": "Finance",
}


def write_fama_french_outputs(
    prices_path: str | Path,
    factors_path: str | Path,
    output_dir: str | Path,
    *,
    annualization: int = 252,
    rolling_window: int = 252,
    sector_map: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run FF3/HAC/GRS diagnostics and write the complete research bundle."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    factors = pd.read_csv(factors_path, index_col=0, parse_dates=True)
    required_factors = ["Mkt-RF", "SMB", "HML", "RF"]
    if prices.empty or any(column not in factors for column in required_factors):
        raise ValueError("invalid stock-price or FF3 factor input")
    returns = prices.pct_change().dropna()
    common = returns.index.intersection(factors.index)
    returns = returns.loc[common]
    factors = factors.loc[common]
    if len(common) <= max(rolling_window, 5):
        raise ValueError("not enough aligned observations for rolling FF3 regression")
    if not returns.index.equals(factors.index) or returns.isna().any().any():
        raise AssertionError("return and factor indexes are not exactly aligned")

    tickers = sorted(prices.columns.astype(str).tolist())
    excess = returns[tickers].subtract(factors["RF"], axis=0)
    factor_matrix = factors[["Mkt-RF", "SMB", "HML"]].to_numpy(float)
    n_obs = len(common)
    design = np.column_stack([np.ones(n_obs), factor_matrix])
    xtx_inv = np.linalg.inv(design.T @ design)
    nw_lag = int(math.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))

    stock_results: dict[str, dict[str, float]] = {}
    alphas: list[float] = []
    residual_matrix = np.zeros((n_obs, len(tickers)), dtype=float)
    reconstruction_error = 0.0
    for column, ticker in enumerate(tickers):
        response = excess[ticker].to_numpy(float)
        beta = np.linalg.lstsq(design, response, rcond=None)[0]
        predicted = design @ beta
        residual = response - predicted
        reconstruction_error = max(
            reconstruction_error, float(np.max(np.abs(response - predicted - residual)))
        )
        alphas.append(float(beta[0]))
        residual_matrix[:, column] = residual
        ss_res = float(residual @ residual)
        ss_tot = float(((response - response.mean()) ** 2).sum())
        r_squared = 1.0 - ss_res / ss_tot
        adjusted = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - 4)
        covariance_ols = (ss_res / (n_obs - 4)) * xtx_inv
        alpha_se = math.sqrt(float(covariance_ols[0, 0]))
        t_alpha = float(beta[0] / alpha_se)
        p_alpha = float(2.0 * stats.t.sf(abs(t_alpha), df=n_obs - 4))

        meat = (design * (residual * residual)[:, None]).T @ design
        for lag in range(1, nw_lag + 1):
            weight = 1.0 - lag / (nw_lag + 1.0)
            products = residual[lag:] * residual[:-lag]
            gamma = (design[lag:] * products[:, None]).T @ design[:-lag]
            meat += weight * (gamma + gamma.T)
        covariance_nw = xtx_inv @ meat @ xtx_inv
        alpha_nw_se = math.sqrt(max(float(covariance_nw[0, 0]), 0.0))
        t_alpha_nw = float(beta[0] / alpha_nw_se)
        p_alpha_nw = float(2.0 * stats.t.sf(abs(t_alpha_nw), df=n_obs - 4))
        annual_alpha = float(beta[0] * annualization)
        annual_excess = float(response.mean() * annualization)
        annual_residual_std = float(residual.std(ddof=1) * math.sqrt(annualization))
        stock_results[ticker] = {
            "alpha_daily": round(float(beta[0]), 8),
            "alpha_annualized": round(annual_alpha, 6),
            "beta_mkt": round(float(beta[1]), 6),
            "beta_smb": round(float(beta[2]), 6),
            "beta_hml": round(float(beta[3]), 6),
            "r_squared": round(r_squared, 6),
            "adj_r_squared": round(adjusted, 6),
            "t_stat_alpha": round(t_alpha, 4),
            "p_value_alpha": round(p_alpha, 6),
            "t_stat_alpha_nw": round(t_alpha_nw, 4),
            "p_value_alpha_nw": round(p_alpha_nw, 6),
            "durbin_watson": round(float((np.diff(residual) ** 2).sum() / ss_res), 6),
            "information_ratio": round(annual_alpha / annual_residual_std, 6),
            "treynor_ratio": round(annual_excess / float(beta[1]), 6),
            "ann_excess_return": round(annual_excess, 6),
        }

    alpha_vector = np.asarray(alphas)
    residual_covariance = residual_matrix.T @ residual_matrix / n_obs
    factor_means = factor_matrix.mean(axis=0)
    demeaned_factors = factor_matrix - factor_means
    factor_covariance = demeaned_factors.T @ demeaned_factors / n_obs
    numerator = float(alpha_vector @ np.linalg.solve(residual_covariance, alpha_vector))
    denominator = 1.0 + float(
        factor_means @ np.linalg.solve(factor_covariance, factor_means)
    )
    grs = (n_obs - len(tickers) - 3) / len(tickers) * numerator / denominator
    grs_p = float(stats.f.sf(grs, len(tickers), n_obs - len(tickers) - 3))

    vif: dict[str, float] = {}
    for factor_index, name in enumerate(("Mkt_RF", "SMB", "HML")):
        response = factor_matrix[:, factor_index]
        others = np.delete(factor_matrix, factor_index, axis=1)
        auxiliary = np.column_stack([np.ones(n_obs), others])
        fitted = auxiliary @ np.linalg.lstsq(auxiliary, response, rcond=None)[0]
        r2 = 1.0 - float(((response - fitted) ** 2).sum()) / float(
            ((response - response.mean()) ** 2).sum()
        )
        vif[name] = round(1.0 / (1.0 - r2), 4)

    rolling_data: dict[str, list[float]] = {}
    for ticker in tickers:
        response = excess[ticker].to_numpy(float)
        values = [
            float(np.linalg.lstsq(
                design[start:start + rolling_window],
                response[start:start + rolling_window], rcond=None,
            )[0][1])
            for start in range(n_obs - rolling_window + 1)
        ]
        rolling_data[ticker] = values
        stock_results[ticker].update(
            rolling_beta_mkt_mean=round(float(np.mean(values)), 6),
            rolling_beta_mkt_std=round(float(np.std(values, ddof=1)), 6),
            rolling_beta_mkt_min=round(float(np.min(values)), 6),
            rolling_beta_mkt_max=round(float(np.max(values)), 6),
        )

    rolling_dates = common[rolling_window - 1:]
    rolling_frame = pd.DataFrame({"date": rolling_dates, **rolling_data})
    alpha_ranking = sorted(
        tickers, key=lambda ticker: stock_results[ticker]["alpha_annualized"], reverse=True
    )
    results = {
        "meta": {
            "trading_days": int(n_obs),
            "start_date": str(common.min().date()),
            "end_date": str(common.max().date()),
            "num_stocks": len(tickers),
            "nw_lag": nw_lag,
        },
        "stocks": stock_results,
        "alpha_ranking": alpha_ranking,
        "grs_test": {
            "grs_statistic": round(grs, 4),
            "grs_p_value": round(grs_p, 6),
            "grs_rejected": bool(grs_p < 0.05),
            "df1": len(tickers),
            "df2": n_obs - len(tickers) - 3,
        },
        "factor_diagnostics": {
            "vif_mkt_rf": vif["Mkt_RF"], "vif_smb": vif["SMB"],
            "vif_hml": vif["HML"], "nw_lag": nw_lag,
        },
    }

    regression_rows = []
    for ticker in tickers:
        values = stock_results[ticker]
        regression_rows.append({
            "ticker": ticker,
            **{key: values[key] for key in (
                "alpha_annualized", "beta_mkt", "beta_smb", "beta_hml",
                "r_squared", "adj_r_squared", "t_stat_alpha", "p_value_alpha",
                "t_stat_alpha_nw", "p_value_alpha_nw", "durbin_watson",
                "information_ratio", "treynor_ratio",
            )},
        })
    regression_frame = pd.DataFrame(regression_rows)

    sectors = dict(DEFAULT_EQUITY_SECTORS if sector_map is None else sector_map)
    if any(ticker not in sectors for ticker in tickers):
        raise ValueError("sector_map must cover every ticker")
    sector_rows = []
    for sector in sorted({sectors[ticker] for ticker in tickers}):
        members = [stock_results[ticker] for ticker in tickers if sectors[ticker] == sector]
        sector_rows.append({
            "sector": sector,
            "avg_alpha_annualized": round(float(np.mean([x["alpha_annualized"] for x in members])), 6),
            "avg_beta_mkt": round(float(np.mean([x["beta_mkt"] for x in members])), 6),
            "avg_r_squared": round(float(np.mean([x["r_squared"] for x in members])), 6),
        })
    sector_frame = pd.DataFrame(sector_rows)

    # Three independent checks: reconstruction, covariance conditions, and ranking.
    if reconstruction_error > 1.0e-14:
        raise AssertionError("OLS prediction and residual do not reconstruct returns")
    if residual_matrix.shape != (n_obs, len(tickers)) or rolling_frame.isna().any().any():
        raise AssertionError("factor output shapes are inconsistent")
    if np.linalg.eigvalsh(factor_covariance).min() <= 0.0:
        raise AssertionError("factor covariance is not positive definite")
    if alpha_ranking != sorted(
        tickers, key=lambda ticker: stock_results[ticker]["alpha_annualized"], reverse=True
    ):
        raise AssertionError("alpha ranking is inconsistent")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_json(target / "results.json", results)
    regression_frame.to_csv(target / "regression_summary.csv", index=False)
    sector_frame.to_csv(target / "sector_summary.csv", index=False)
    rolling_frame.to_csv(target / "rolling_betas.csv", index=False)

    positions = np.arange(len(tickers)); width = 0.25
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.bar(positions - width, [stock_results[t]["beta_mkt"] for t in tickers], width, label="Mkt-RF")
    axis.bar(positions, [stock_results[t]["beta_smb"] for t in tickers], width, label="SMB")
    axis.bar(positions + width, [stock_results[t]["beta_hml"] for t in tickers], width, label="HML")
    axis.set_xticks(positions); axis.set_xticklabels(tickers, rotation=45)
    axis.set_title("Fama-French 3-Factor Loadings"); axis.legend(); axis.axhline(0, color="black", linewidth=0.5)
    figure.tight_layout(); figure.savefig(target / "factor_loadings.png", dpi=100); plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 7))
    ranked_values = [stock_results[t]["alpha_annualized"] for t in alpha_ranking]
    labels = [f"{t} *" if stock_results[t]["p_value_alpha_nw"] < 0.05 else t for t in alpha_ranking]
    axis.barh(range(len(tickers)), ranked_values, color=["green" if x > 0 else "red" for x in ranked_values])
    axis.set_yticks(range(len(tickers))); axis.set_yticklabels(labels); axis.invert_yaxis()
    axis.set_title("Annualized Jensen's Alpha (NW significance)"); axis.axvline(0, color="black", linewidth=0.5)
    figure.tight_layout(); figure.savefig(target / "alpha_chart.png", dpi=100); plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(tickers, [stock_results[t]["r_squared"] for t in tickers])
    axis.set_ylim(0, 1); axis.set_title("R-squared — Fama-French 3-Factor Model")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout(); figure.savefig(target / "r_squared_chart.png", dpi=100); plt.close(figure)

    figure, axis = plt.subplots(figsize=(14, 8))
    for ticker in tickers:
        axis.plot(rolling_dates, rolling_data[ticker], label=ticker, alpha=0.7)
    axis.set_title("Rolling 252-Day Market Beta"); axis.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    figure.tight_layout(); figure.savefig(target / "rolling_betas.png", dpi=100); plt.close(figure)

    return {
        "results": results, "regression_summary": regression_frame,
        "sector_summary": sector_frame, "rolling_betas": rolling_frame,
    }
