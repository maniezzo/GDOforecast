import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import json
from statsmodels.tsa.ar_model import AutoReg
from scipy.signal import lfilter
from statsmodels.regression.linear_model import yule_walker

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

def read_TPdata():
   # legge i costi
   dfcost = pd.read_csv("costs.csv")
   cost = np.array(dfcost.values)
   # legge le capacita
   cap = pd.read_csv("capacities.csv",header=None).values.flatten()
   # calcola le richieste per ogni bootstrap (scenario)
   req = []
   for i in range(df.shape[1]):
      ds = df.iloc[:,i]
      x = ds[:-3].to_numpy()
      
      # prewhitening, forecasting, recoloring and getting the 3mh point forecast
      pwx, params = prewhiten(x)    # prewhiten
      f = ar_rolling_forecast(pwx,1,3,"n")
      f48 = np.append(pwx,f)
      x_back = recolor(f48, params) # recolr
      #print(f"serie {i} forecast {x_back[-3:]} <-> true {ds[-3:].to_numpy()}")
      r = np.ceil(x_back[-1])
      req.append(r)
   return cost,cap,np.array(req).flatten()

# solve transportation problem
from scipy.optimize import linprog
from scipy.sparse import kron, eye, csr_matrix
def compute_transport():
   cost, cap, req = read_TPdata()
   
   """Min-cost transportation: servers (rows) with capacity cap,
   clients (columns) with request req. Returns flow matrix and total cost."""
   #cost, cap, req = map(np.asarray, (cost, cap, req))
   m, n = cost.shape
   if cap.sum() < req.sum():
      raise ValueError("infeasible: total capacity < total requests")
   
   ones_n = csr_matrix(np.ones((1, n)))
   ones_m = csr_matrix(np.ones((1, m)))
   A_ub = kron(eye(m), ones_n)  # each server: sum_j x_ij <= cap_i
   A_eq = kron(ones_m, eye(n))  # each client: sum_i x_ij == req_j
   
   res = linprog(cost.ravel(), A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=req,
                 bounds=(0, None), method="highs")
   if not res.success:
      raise RuntimeError(res.message)
   return res.x.reshape(m, n), res.fun
   

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

if __name__ == '__main__':
   costs = pd.read_csv("costs.csv").values
   cap   = pd.read_csv("capacities.csv",header=None).values.flatten()

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   print(f"Boost forecasting {name}")

   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   read_TPdata()
   flow, total = compute_transport()
   print(flow)
   print("total cost:", total)

   print('Fine')
