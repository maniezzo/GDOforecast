import numpy as np
from scipy.signal import lfilter
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.ar_model import AutoReg

def prewhiten(x, s=12, p=2):
   #Remove linear trend + seasonal means, then whiten with AR(p)
   t = np.arange(len(x))
   trend = np.polyfit(t, x, 1)  # linear trend coefficients
   r = x - np.polyval(trend, t)
   seas = np.array([r[t % s == k].mean() for k in range(s)])
   r = r - seas[t % s]  # deseasoned residual
   phi = yule_walker(r, order=p, method="mle")[0]
   a = np.r_[1.0, -phi]  # AR polynomial A(z)
   e = lfilter(a, [1.0], r)  # white residuals
   return e, (trend, seas, a)

def recolor(e, params):
   #inverse of prewhiten: undo AR, add seasonality and trend back
   trend, seas, a = params
   t = np.arange(len(e))
   r = lfilter([1.0], a, e)
   return r + seas[t % len(seas)] + np.polyval(trend, t)


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
