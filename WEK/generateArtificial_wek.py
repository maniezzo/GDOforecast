"""Generate ETS request scenarios rho for the WEK Wasserstein model.

Rows of rho are equally weighted joint scenarios; columns are customers.
Artificial customers are sampled with replacement from source dataframe
columns. Each ETS forecast targets month 48, three steps after the 45-month
training history.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint
from statsmodels.tsa.exponential_smoothing.ets import ETSModel


def generate_artificial(
    numCustomers, numCenters, numscen, *, demand_scale=1.0, seed=995,
    costs_path="costs.csv", capacities_factor=1.1, capacities_power=1.1,
):
    """Return ``rho, c, h, Q, test_values`` for one generated instance.

    ``rho`` has shape (numscen, numCustomers); ``c`` has shape
    (numCenters, numCustomers); ``h`` and ``Q`` have shape (numCenters,);
    ``test_values`` contains each sampled customer's observed row-48 value.
    Fixed assignment costs ``c`` are read from ``costs_path``. Capacities
    ``Q`` and unit storage costs ``h`` are recalculated from this generation's
    point forecasts and distance submatrix.
    """
    name = "dataframe_nocovid_full"
    dfseries = pd.read_csv(f"../data/{name}.csv", usecols=range(1, 53))
    history, test_values = sample_customer_series(
        dfseries, n_new_cols=numCustomers, random_state=seed
    )
    rho, _, ets_info = ets_saa_scenarios(
        history.to_numpy(), n_scenarios=numscen, horizon=3,
        demand_scale=demand_scale, trend="add", damped_trend=True,
        seasonal_periods=12, seed=seed,
    )

    c_all = pd.read_csv(costs_path).to_numpy(dtype=float)
    if (c_all.ndim != 2 or c_all.shape[0] < numCenters
            or c_all.shape[1] < numCustomers):
        raise ValueError(
            f"Cost matrix must have at least {numCenters} center rows and "
            f"{numCustomers} customer columns."
        )
    c = c_all[:numCenters, :numCustomers].copy()

    # Preserve the original artificial-instance construction: capacities
    # are based on total point-forecast demand, while h uses nearest-center
    # distances and those capacities.
    point_req = np.ceil(ets_info["mean"] * demand_scale).astype(int)
    Q = np.asarray(generate_capacities(
        point_req.sum(), numCenters,
        factor=capacities_factor, power=capacities_power,
    )[::-1], dtype=float)
    dist_all = pd.read_csv("../matrice0.csv", header=None).to_numpy(dtype=float)
    dist = dist_all[numCenters:numCenters + numCustomers, :numCenters]
    if dist.shape != (numCustomers, numCenters):
        raise ValueError("Distance matrix dimensions do not match customers and centers.")
    h = solve_h(np.min(dist, axis=0), Q)
    return rho, c, h, Q, test_values

def sample_customer_series(df, n_new_cols, random_state=None):
    """Sample columns with replacement; return 45 training rows and row-48 truth."""
    if df.shape[0] < 48:
        raise ValueError("Expected at least 48 monthly observations in the source dataframe.")
    rng = np.random.default_rng(random_state)
    sampled_cols = rng.choice(df.columns, size=n_new_cols, replace=True)
    sampled = df.loc[:, sampled_cols]
    test_values = sampled.iloc[47].to_numpy()
    history = sampled.iloc[:45].copy()
    history.columns = [f"customer{i}" for i in range(n_new_cols)]
    return history, test_values


def ets_saa_scenarios(history, n_scenarios, horizon=3, *, demand_scale=1.0,
                      trend="add", damped_trend=True, seasonal_periods=12,
                      alpha=0.05, seed=995):
    """Fit one additive ETS model per customer and draw endpoint scenarios.

    The Gaussian approximation is calibrated from the ETS prediction
    interval. Negative draws are clipped at zero and demands are rounded up.
    Returns integer scenarios (scenario x customer), uniform probabilities,
    and per-customer predictive means.
    """
    y = np.asarray(history, dtype=float)
    if y.ndim != 2 or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("history must be a finite, nonnegative (time x customer) array.")
    if n_scenarios < 1 or horizon < 1:
        raise ValueError("n_scenarios and horizon must be positive.")
    T, n = y.shape
    scale = np.broadcast_to(np.asarray(demand_scale, dtype=float), (n,))
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("demand_scale must be finite and positive.")

    means = np.empty(n)
    stds = np.empty(n)
    zcrit = norm.ppf(1 - alpha / 2)
    for j in range(n):
        print(f"ETS forecast for customer {j}")
        series = y[:, j]
        if np.ptp(series) == 0:
            means[j] = series[0]
            stds[j] = 0.0
            continue
        fit = ETSModel(
            pd.Series(series), error="add", trend=trend,
            damped_trend=bool(damped_trend and trend is not None),
            seasonal="add" if seasonal_periods is not None else None,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        ).fit(disp=False, maxiter=2000)
        if not fit.mle_retvals.get("converged", True):
            raise RuntimeError(f"ETS fit did not converge for customer {j}.")
        frame = fit.get_prediction(
            start=T, end=T + horizon - 1, method="exact"
        ).summary_frame(alpha=alpha)
        endpoint = frame.iloc[-1]
        means[j] = endpoint["mean"]
        stds[j] = (endpoint["pi_upper"] - endpoint["pi_lower"]) / (2 * zcrit)

    rng = np.random.default_rng(seed)
    monthly = means + stds * rng.standard_normal((n_scenarios, n))
    scenarios = np.ceil(np.maximum(monthly, 0.0) * scale).astype(np.int64)
    probabilities = np.full(n_scenarios, 1.0 / n_scenarios)
    return scenarios, probabilities, {"mean": means, "std": stds}




def generate_capacities(sumreq, numCenters, factor=1.1, power=1.2):
    """Generate integer capacities with total approximately factor * sumreq."""
    if numCenters < 1:
        raise ValueError("numCenters must be positive.")
    target = round(factor * sumreq)
    weights = np.array([(i + 1) ** power for i in range(numCenters)], dtype=float)
    capacities = np.floor(target * weights / weights.sum()).astype(int)
    capacities[-1] += target - capacities.sum()
    return capacities.tolist()


def solve_h(d, req):
    """Reproduce the original decreasing storage-cost calibration."""
    d = np.asarray(d, dtype=float)
    req = np.asarray(req, dtype=float)
    if d.ndim != 1 or d.shape != req.shape:
        raise ValueError("d and req must be 1-D arrays of equal length.")
    if np.any(d <= 0) or np.any(req <= 0):
        raise ValueError("d and req must be positive.")
    order = np.argsort(d, kind="stable")
    ds, rs = d[order], req[order]
    n = len(ds)
    h_min = ds[0] / rs[0]
    target_mean = d.mean()
    if n == 1:
        return np.array([h_min])
    w = rs
    inv_d = 1.0 / ds
    lower = h_min * w.mean()
    if target_mean < lower * (1 - 1e-12):
        raise ValueError(
            f"Infeasible: mean(d) = {target_mean:.6g} is below "
            f"h_min * mean(req) = {lower:.6g}."
        )

    def objective(x):
        residual = x[:n] - x[n] * inv_d
        return residual @ residual, np.r_[2 * residual, -2 * (residual @ inv_d)]

    A_eq = np.zeros((2, n + 1))
    A_eq[0, :n] = w / n
    A_eq[1, n - 1] = 1.0
    b_eq = np.array([target_mean, h_min])
    A_mono = np.zeros((n - 1, n + 1))
    idx = np.arange(n - 1)
    A_mono[idx, idx] = 1.0
    A_mono[idx, idx + 1] = -1.0
    constraints = [
        LinearConstraint(A_eq, b_eq, b_eq),
        LinearConstraint(A_mono, 0.0, np.inf),
    ]
    alpha0 = h_min * ds[-1]
    x0 = np.r_[alpha0 * inv_d, alpha0]
    result = minimize(
        objective, x0, jac=True, method="SLSQP",
        bounds=[(0.0, None)] * (n + 1), constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not result.success:
        raise RuntimeError(f"Storage-cost optimization failed: {result.message}")
    h = np.empty(n)
    h[order] = result.x[:n]
    return h
