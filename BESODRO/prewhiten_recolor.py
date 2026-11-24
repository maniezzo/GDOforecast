"""Linear trend + multiplicative seasonality, with optional AR forecasting.
This removes deterministic structure, not serial dependence.
The returned z will be an input to an AR model.
No logarithm of y is taken. Assumptions: regularly sampled positive data,
an approximately linear trend, and a repeating seasonal multiplier profile.
    relative=True:  y[t] = (a + b*t) * S[t % period] * (1 + z[t])
    relative=False: y[t] = (a + b*t) * S[t % period] + z[t]

Use relative residuals when remainder fluctuations scale with the baseline;
use absolute residuals when their magnitude is approximately constant.
Stationarity and forecast quality must be assessed on the actual data.
Fit the decomposition and AR afresh at each training origin in validation.
"""
import numpy as np
from scipy.optimize import least_squares
from statsmodels.tsa.ar_model import AutoReg

def _vector(values, name):
   values = np.asarray(values, dtype=float)
   if values.ndim != 1 or not np.all(np.isfinite(values)):
      raise ValueError(f"{name} must be a finite, one-dimensional array.")
   return values


def _integer(value, name, minimum):
   if (isinstance(value, (bool, np.bool_)) or
         not isinstance(value, (int, np.integer)) or value < minimum):
      raise ValueError(f"{name} must be an integer >= {minimum}.")

# computes the trend × seasonality component for a specified range of time positions
def _baseline(model, start, length):
   _integer(start, "start", 0)
   t = start + np.arange(length)
   trend = model["level"] + model["slope"] * t
   if np.any(trend <= 0) or not np.all(np.isfinite(trend)):
      raise ValueError("The extrapolated linear trend is not positive and finite.")
   return trend * model["seasonal"][t % model["period"]]


def prewhiten(y, period=12, relative=True):
   """Fit the deterministic baseline and return (z, model).

   The linear trend and positive seasonal multipliers are estimated JOINTLY
   by nonlinear least squares on the original scale, minimizing
   sum((y[t] - (a+b*t)*S[t % period])**2).

   Positive trend endpoints and positive seasonal coefficients are enforced.
   Seasonal multipliers are normalized to have mean 1. At least two seasonal
   cycles are required; substantially more data gives more reliable estimates.
   All len(y) observations are retained in z. The seasonal phase at y[0]
   defines phase zero, so dates are not required for regular data.

   The objective is original-scale least squares for both remainder modes.
   It is not a maximum-likelihood fit of a joint AR/multiplicative-error model.
   """
   y = _vector(y, "y")
   _integer(period, "period", 2)
   n = len(y)
   if n < 2 * period:
      raise ValueError("Supply at least two complete seasonal cycles.")
   if np.any(y <= 0):
      raise ValueError("y must be strictly positive.")
   
   # Normalize to improve numerical conditioning; time runs from 0 to 1.
   scale = float(np.mean(y))
   yn = y / scale
   x = np.linspace(0.0, 1.0, n)
   phase = np.arange(n) % period
   
   # Initial trend from complete-cycle means, reducing seasonal confounding.
   cycles = n // period
   cycle_y = yn[:cycles*period].reshape(cycles, period).mean(axis=1)
   cycle_x = (period*np.arange(cycles) + (period-1)/2) / (n-1)
   initial_line = np.linalg.lstsq(
      np.column_stack([np.ones(cycles), cycle_x]), cycle_y, rcond=None
   )[0]
   endpoints = np.maximum([initial_line[0], initial_line.sum()], 1e-4)
   initial_trend = endpoints[0]*(1-x) + endpoints[1]*x
   ratios = yn / initial_trend
   seasonal = np.array([ratios[phase == j].mean() for j in range(period)])
   
   # Fix S[0]=1 during fitting to remove the trend/season scale ambiguity.
   theta0 = np.r_[endpoints*seasonal[0], seasonal[1:]/seasonal[0]]
   
   def residual(theta):
      trend = theta[0]*(1-x) + theta[1]*x
      s = np.r_[1.0, theta[2:]]
      return trend*s[phase] - yn
   
   fit = least_squares(
      residual, theta0, bounds=(1e-12, np.inf), x_scale="jac",
      max_nfev=5000, ftol=1e-11, xtol=1e-11, gtol=1e-11
   )
   if not fit.success:
      raise RuntimeError(f"Trend/seasonal fit failed: {fit.message}")
   
   seasonal = np.r_[1.0, fit.x[2:]]
   seasonal_mean = seasonal.mean()
   model = {
      "level": float(fit.x[0]*seasonal_mean*scale),
      "slope": float((fit.x[1]-fit.x[0])*seasonal_mean*scale/(n-1)),
      "seasonal": seasonal / seasonal_mean,
      "period": int(period),
      "n_train": n,
      "relative": bool(relative),
   }
   baseline = _baseline(model, 0, n)
   z = y / baseline - 1.0 if relative else y - baseline
   return z, model


def recolor(z, model, *, start=0):
   """Restore historical values or forecasts to the original scale.

   For historical reconstruction, use start=0 (default).
   For forecasts immediately after training, use start=model['n_train'].
   More generally start is a zero-based index on the ORIGINAL time axis.
   Future trend is extrapolated linearly and seasonal multipliers repeat.

   An AR forecast can violate positivity; this raises instead of clipping.
   No AR filtering or integration is done here; z must already be forecast.
   """
   z = _vector(z, "z")
   baseline = _baseline(model, start, len(z))
   y = baseline*(1.0 + z) if model["relative"] else baseline + z
   if np.any(y <= 0) or not np.all(np.isfinite(y)):
      raise ValueError("Recolored values are not positive and finite; inspect the AR forecast.")
   return y

# not used, I am using AutoReg instead
def ar_forecast(z, horizon, p=1):
   """Simple AR(p) point forecast, fitted by OLS with an intercept.
   Optional convenience function: your existing AR implementation can replace
   it. No AR-stability constraint is imposed. p=0 forecasts the sample mean.
   Choose p and assess forecasting performance on training/validation data.
   """
   z = _vector(z, "z")
   _integer(horizon, "horizon", 1)
   _integer(p, "p", 0)
   n = len(z)
   if n <= 2*p + 1:
      raise ValueError("Need len(z) > 2*p + 1; reduce p or supply more data.")
   X = np.column_stack([np.ones(n-p)] +
                       [z[p-k:n-k] for k in range(1, p+1)])
   beta = np.linalg.lstsq(X, z[p:], rcond=None)[0]
   extended = np.empty(n+horizon)
   extended[:n] = z
   for t in range(n, n+horizon):
      extended[t] = beta[0] + beta[1:] @ extended[t-p:t][::-1]
   return extended[n:]

if __name__ == "__main__":
   rng = np.random.default_rng(123)
   t = np.arange(120)
   seasonal = 1 + 0.25*np.sin(2*np.pi*t/12)
   remainder = np.zeros(len(t))
   for k in range(1, len(t)):
      remainder[k] = 0.6*remainder[k-1] + rng.normal(scale=0.025)
   y = (100 + 0.8*t)*seasonal*(1+remainder)
   
   z, model = prewhiten(y, period=12)
   np.testing.assert_allclose(recolor(z, model), y, rtol=1e-12, atol=1e-12)
   #z_hat = ar_forecast(z, horizon=3, p=1)
   fit = AutoReg(z, lags=1, trend="c", seasonal=False).fit()
   z_hat = fit.forecast(steps=3)
   
   y_hat = recolor(z_hat, model, start=len(y))
   print("Maximum reconstruction error:", np.max(np.abs(recolor(z, model)-y)))
   print("Next three forecasts:", y_hat)
