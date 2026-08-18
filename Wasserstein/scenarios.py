'''
Build AR-based empirical scenarios without bootstrapping
The important point is to preserve the joint error vector across stores.
At each rolling forecast origin, generate forecast errors for all 52 stores
at the same origin; don't independently mix residuals between stores.
'''
import numpy as np, pandas as pd
from statsmodels.tsa.ar_model import AutoReg

# genera scenari AR dato il dataframe con le vendite
def go_scenarios():
   df = pd.read_csv("../data/dataframe_nocovid_full.csv").iloc[:, 1:]
   scenarios, errors, current_forecast = build_ar_scenarios(np.array(df))
   return df, scenarios

# AR(p) rolling
def ar_rolling_forecast(y, p, n_forecast, trend="c"):
    """
    Rolling multi-step forecast using an AR(p) model.
    Parameters
        y : array-like,         Observed time series.
        p : int,         Autoregressive order.
        n_forecast : int,         Number of successive future observations to forecast.
        trend : str, default="c". Deterministic terms: "n" = none, "c" = constant, "ct" = constant and trend.
    Returns
        numpy.ndarray: Array containing n_forecast predictions.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim != 1: raise ValueError("y must be one-dimensional")
    if p < 1:       raise ValueError("p must be at least 1")
    if n_forecast < 1: raise ValueError("n_forecast must be at least 1")
    if len(y) <= p:    raise ValueError("The series must contain more observations than p")

    history = list(y)
    forecasts = []

    for _ in range(n_forecast):
        model = AutoReg(history, lags=p, trend=trend, old_names=False).fit()
        next_value = model.predict(start=len(history), end=len(history) )[0]
        forecasts.append(next_value)
        history.append(next_value)

    return np.asarray(np.ceil(forecasts)) # numero di pallet intero

# ON-BOOTSTRAPPED empirical predictive scenarios from  AR forecast errors
# gli scenariu saranno len(Y)-min_train
def build_ar_scenarios(dfdata, min_train=24):
    #  Empirical scenarios for demand exactly 3 periods ahead.
    Y = np.asarray(dfdata, dtype=float)[:-3,:] # tolgo gli ultimi 3 dati
    T, J = Y.shape
    H = 3 # horizon

    # Current h=3 point forecast
    current_forecast = np.zeros(J)

    for j in range(J):
        pred = np.asarray(ar_rolling_forecast(Y[:, j],6,H))
        current_forecast[j] = pred[H - 1]

    errors = []
    # Historical rolling forecast origins
    for t in range(min_train, T - H + 1):
        e = np.zeros(J)
        for j in range(J):
            train = Y[:t, j]
            pred = np.asarray(ar_rolling_forecast(Y[:, j],6,H))

            # pred[2] is the 3-step-ahead forecast
            forecast_h3 = pred[H - 1]

            # With this indexing, Y[t] is h=1,
            # Y[t+1] is h=2, Y[t+2] is h=3
            actual_h3 = Y[t + H - 1, j]

            e[j] = actual_h3 - forecast_h3

        errors.append(e)

    errors = np.asarray(errors)

    # Shift historical h=3 forecast errors onto today's h=3 forecast
    scenarios = current_forecast[None, :] + errors

    # if pallets must be nonnegative integers
    scenarios = np.maximum(scenarios, 0.0)
    scenarios = np.rint(scenarios)

    return scenarios, errors, current_forecast
