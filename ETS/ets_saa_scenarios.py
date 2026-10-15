"""Non-bootstrap Gaussian ETS scenarios for the BESO SAA interface.

Dependencies: numpy, scipy, statsmodels.
Input: ORIGINAL monthly demand levels, time x stores, training data only.
Output: nonnegative integer demands, scenarios x stores; uniform probabilities.
The horizon is a single future month, NOT cumulative demand over that horizon.
Fitted parameters are treated as fixed: parameter uncertainty is not simulated.
"""

import warnings
import numpy as np, pandas as pd
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

def ets_saa_scenarios(
    history, n_scenarios=125, horizon=3, *,
    demand_scale, trend="add", damped_trend=True,
    seasonal_periods=None, alpha=0.05, seed=12345,
    correlation=None,
):
    """Fit additive ETS per store and sample its endpoint predictive law.
    history : (T, n) array of original monthly demand levels.
        Do not pass prewhitened/differenced residuals.
    demand_scale : positive scalar or (n,) array.
        Explicit conversion from monthly demand units to the optimization's
        two-week pallet units. Use exactly the same conversion as in BESO.
        Use 1.0 only when no conversion is needed.
    seasonal_periods : None or int.
        None: nonseasonal ETS. 12: additive annual seasonality in monthly data.
    correlation : None or (n, n) predictive-error correlation matrix.
        None assumes independent store forecasts. A supplied matrix must be
        symmetric, positive semidefinite, with unit diagonal. It should describe
        errors at the requested horizon, estimated from training data only.

    Returns
    -------
    rho : (n_scenarios, n) int64 array
    p : (n_scenarios,) array, all entries 1/n_scenarios
    info : dict
        Monthly-scale mean, std, lower, upper BEFORE clipping/rounding;
        fits; observed negative-draw fractions; dependence assumption.
    """
    y = np.asarray(history, dtype=float)
    if y.ndim != 2 or y.shape[0] < 10 or y.shape[1] < 1:
        raise ValueError("history must have shape (T, n), T >= 10, n >= 1.")
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("history must contain finite nonnegative demand levels.")
    for name, value in [("n_scenarios", n_scenarios), ("horizon", horizon)]:
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    if trend not in (None, "add"):
        raise ValueError("trend must be None or 'add'.")
    if seasonal_periods is not None:
        if (not isinstance(seasonal_periods, (int, np.integer))
                or seasonal_periods < 2 or y.shape[0] < 2 * seasonal_periods):
            raise ValueError("Seasonality requires at least two complete cycles.")
    T, n = y.shape
    scale = np.broadcast_to(np.asarray(demand_scale, dtype=float), (n,))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("demand_scale must be finite and positive.")
    mu, sigma, lower, upper = (np.empty(n) for _ in range(4))
    fits = []
    zcrit = norm.ppf(1 - alpha / 2)
    for j in range(n):
        series = y[:, j]
        if np.ptp(series) == 0:
            # Explicit deterministic convention for constant histories.
            mu[j] = lower[j] = upper[j] = series[0]
            sigma[j] = 0.0
            fits.append(None)
            warnings.warn(f"Store {j}: constant history; zero forecast variance.")
            continue
        fit = ETSModel(
            pd.Series(series, index=pd.RangeIndex(T)),
            error="add", trend=trend,
            damped_trend=bool(damped_trend and trend is not None),
            seasonal="add" if seasonal_periods is not None else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(disp=False, maxiter=2000)
        if not fit.mle_retvals.get("converged", True):
            raise RuntimeError(f"ETS fit did not converge for store {j}.")
        # end is inclusive: T + horizon - 1 is the horizon-th future month.
        frame = fit.get_prediction(
            start=T, end=T + horizon - 1, method="exact"
        ).summary_frame(alpha=alpha)
        last = frame.iloc[-1]
        mu[j] = last["mean"]
        lower[j], upper[j] = last["pi_lower"], last["pi_upper"]
        sigma[j] = (upper[j] - lower[j]) / (2 * zcrit)
        fits.append(fit)
    if not np.isfinite([mu, sigma, lower, upper]).all() or (sigma < 0).any():
        raise RuntimeError("ETS returned invalid predictive moments.")
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_scenarios, n))
    if correlation is not None:
        R = np.asarray(correlation, dtype=float)
        if (R.shape != (n, n) or not np.isfinite(R).all()
                or not np.allclose(R, R.T, atol=1e-8)
                or not np.allclose(np.diag(R), 1.0, atol=1e-8)):
            raise ValueError("Invalid predictive-error correlation matrix.")
        eigenvalues, eigenvectors = np.linalg.eigh((R + R.T) / 2)
        if eigenvalues.min() < -1e-8:
            raise ValueError("correlation must be positive semidefinite.")
        root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        z = z @ root.T
    monthly = mu + sigma * z
    negative_fraction = (monthly < 0).mean(axis=0)
    # Censor negatives at zero; this is NOT a truncated-normal draw.
    scaled = np.maximum(monthly, 0.0) * scale
    if not np.isfinite(scaled).all() or (scaled >= np.iinfo(np.int64).max).any():
        raise OverflowError("Demand scenarios exceed supported integer range.")
    rho = np.ceil(scaled).astype(np.int64)
    p = np.full(n_scenarios, 1.0 / n_scenarios)
    info = dict(mean=mu, std=sigma, lower=lower, upper=upper, fits=fits,
                negative_fraction=negative_fraction,
                independent_stores=correlation is None)
    return rho, p, info
