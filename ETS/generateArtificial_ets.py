"""Generate artificial BESO instances using ETS forecast scenarios.
Artificial customers are sampled with replacement from the columns of the
source dataframe. For each selected customer, ETS is fitted to the first 45
observations and scenarios are drawn for month 48 (three months ahead).
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from scipy.stats import norm

def go_artificial(numCustomers, numCenters, numscen):
    name = "dataframe_nocovid_full"
    dfseries = pd.read_csv(f"../data/{name}.csv", usecols=range(1, 53))
    history, trueVal = sample_customer_series(
        dfseries, n_new_cols=numCustomers, random_state=995
    )

    # Distances: rows are customers, columns are centers.
    dist = pd.read_csv("../matrice0.csv", header=None).values
    dist = dist[numCenters:numCenters + numCustomers, :numCenters]

    # Each column is one customer's ETS predictive scenarios. The point
    # forecast (ETS predictive mean) is used to size capacities and costs.
    scenarios, p, ets_info = ets_saa_scenarios(
        history.to_numpy(), n_scenarios=numscen, horizon=3,
        demand_scale=1.0, trend="add", damped_trend=True,
        seasonal_periods=12, seed=995,
    )
    fcast = [scenarios[:, i].copy() for i in range(numCustomers)]
    req = np.ceil(ets_info["mean"]).astype(int).tolist()

    print("Defining servers' capacities and costs ...")
    cap = generate_capacities(sum(req), numCenters, factor=1.1, power=1.1)
    cap = cap[::-1]
    print(f"Capacities {cap}")

    h = solve_h(np.min(dist, axis=0), cap, k=0.33).flatten()
    print(f"Inventory costs {h}")

    # Equal scenario weights: objective-aware reweighting is not applied here.
    pOA = np.full(numscen, 1.0 / numscen)
    return scenarios, fcast, h, pOA, dist, cap, trueVal


def sample_customer_series(df, n_new_cols, random_state=None):
    """Sample source columns with replacement; return train history and truth.

    The first 45 rows are the training histories. Row 48 (index 47) is
    retained as the observed value for evaluation, matching the source data
    convention used by the original generator.
    """
    if df.shape[0] < 48:
        raise ValueError("Expected at least 48 monthly rows in the source dataframe.")
    rng = np.random.default_rng(random_state)
    sampled_cols = rng.choice(df.columns, size=n_new_cols, replace=True)
    sampled = df.loc[:, sampled_cols].copy()
    trueval = sampled.iloc[47].to_numpy()
    history = sampled.iloc[:45].copy()
    history.columns = [f"customer{i}" for i in range(n_new_cols)]
    return history, trueval


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

# capacita': esponenziali con somma data (factor*somma richieste)
def generate_capacities(sumreq, numCenters, factor=1.1, power=1.2):
    target = round(factor * sumreq)
    weights = np.array([(i + 1) ** power for i in range(numCenters)], dtype=float)
    capacities = np.floor(target * weights / weights.sum()).astype(int)
    capacities[-1] += target - capacities.sum()
    return capacities.tolist()


def solve_h(d, cap, k=0.33):
    """Find decreasing inventory costs w.r.t. distances.
    d: array of distances
    cap: array of capacities
    k: control param, the higher the further from mirror linear
    Returns
    -------
    h: array of unit handling costs
    """
    d = np.asarray(d, dtype=float)
    cap = np.asarray(cap, dtype=float)
    if d.ndim != 1 or d.shape != cap.shape:
        raise ValueError("d and req must be 1-D arrays of equal length.")
    if np.any(d <= 0) or np.any(cap <= 0) or (k<=0):
        raise ValueError("d, k and req must be positive.")
    D = d.mean()
    h = (D - k * (d - D)) / cap
    return h
