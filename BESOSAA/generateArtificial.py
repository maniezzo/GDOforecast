import numpy as np, pandas as pd
from MEBbootstrap import meboot
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import linprog
from scipy.optimize import minimize, LinearConstraint
from prewhiten_recolor import prewhiten,recolor

def go_artificial(numCustomers,numCenters,numscen,gamma=1):
   name = "dataframe_nocovid_full"
   # train series
   dfseries = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)]).iloc[:45,:]
   df2 = add_bootstrap_columns(dfseries, n_new_cols=numCustomers, random_state=995)
   print(df2.head())
   # true (unknown) values
   testVal = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)]).iloc[-1,:].values

   # distanze: prime righe/colunne centri, poi clienti
   dist = pd.read_csv(f"../matrice0.csv",header=None).values
   dist = dist[numCenters:(numCenters+numCustomers),:numCenters]
   
   # bootstrapping
   boots=[]
   fcast=[]
   req = [] # bagged bootstrap forecasts
   for i in range(numCustomers):
      print(f"Bootsrapping client {i}")
      b = meboot(df2.iloc[:,i],reps=numscen)
      f = []
      req.append(0)
      # AR(p) forecsts
      for col in range(numscen):
         y = b[:,col]  # the col'th bootstrapped series
         z, model = prewhiten(y) # z is the prewhitened bootstrapped series
         fit = AutoReg(z, lags=5, trend="c", seasonal=False).fit()
         z_hat = fit.forecast(steps=3) # next 3, prewhitened
         y_hat = recolor(z_hat, model, start=len(y)) # next 3, original
         # third-step-ahead point forecast
         fc = y_hat[-1]
         fc = np.round(fc,0)
         
         f.append(fc)
         req[i] += fc
         
      req[i] = np.ceil(req[i]/numscen) # previsione assoluta bootstrap
      boots.append(b)                  # serie bootstrappate
      fcast.append(np.array(f))        # previsioni serie bootstrap
      
   print("Defining servers' capacities and costs ...")
   # definizione capacità center note le richieste
   cap = generate_capacities(sum(req), numCenters, factor=1.1, power=1.2)
   cap = cap[::-1]  # reverse values
   print(f"Capacities {cap}")
   
   # calcolo dei costi h di inventory, base min delle distanze per customer
   h = solve_h(np.min(dist, axis=0), cap).flatten()
   print(f"Inventory costs {h}")
   
   # calcolo g_s
   print("Computing Objective-aware probabilities ...")
   gs = []
   sumgs=0
   for s in range(numscen):
      req = []
      for j in range(numCustomers):
         req.append(fcast[j][s])
      gs.append(-1)
      _, gs[s] = solve_transportation(dist.T, req, cap)
      sumgs += gamma*gs[s]
      #print(f"Transportation cost: {gs[s]}")
   pOA = [gamma*gs[s]/sumgs for s in range(numscen)]
   return boots,fcast,h,pOA,dist,cap

# estrae con replacement n_new_cols serie da df, per creare istanze artificiali
def add_bootstrap_columns(df, n_new_cols, random_state=None):
   rng = np.random.default_rng(random_state)
   
   # sample column names with replacement
   sampled_cols = rng.choice(df.columns, size=n_new_cols, replace=True)
   
   # create new dataframe with sampled columns
   new_df = df[sampled_cols].copy()
   
   # rename generated columns
   new_df.columns = [f"boot{i}" for i in range(n_new_cols)]
   
   # append to original dataframe
   return new_df

# problema del trasporto (per g_s)
def solve_transportation(costs, requests, capacities):
    """
    Parameters
    costs : 2D array-like (m x n)    Transportation costs.
    requests : 1D array-like (n)     Demand of each destination.
    capacities : 1D array-like (m)   Supply/capacity of each source.

    Returns
    allocation : ndarray (m x n)   Optimal shipment quantities.
    total_cost : float             Optimal transportation cost.
    """

    costs = np.asarray(costs, dtype=float)
    requests = np.asarray(requests, dtype=float)
    capacities = np.asarray(capacities, dtype=float)

    m, n = costs.shape

    if len(capacities) != m:
        raise ValueError("Capacity vector length must equal number of rows in costs")

    if len(requests) != n:
        raise ValueError("Request vector length must equal number of columns in costs")

    if capacities.sum() < requests.sum():
        raise ValueError(
            f"Total capacity ({capacities.sum()}) is smaller than total demand ({requests.sum()})"
        )

    # Objective vector
    c = costs.flatten()

    # Demand constraints: sum_i x_ij = request_j
    A_eq = np.zeros((n, m * n))
    for j in range(n):
        for i in range(m):
            A_eq[j, i * n + j] = 1
    b_eq = requests

    # Capacity constraints: sum_j x_ij <= capacity_i
    A_ub = np.zeros((m, m * n))
    for i in range(m):
        for j in range(n):
            A_ub[i, i * n + j] = 1
    b_ub = capacities

    # Nonnegative variables
    bounds = [(0, None)] * (m * n)

    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs"
    )

    if not result.success:
        raise RuntimeError(result.message)

    allocation = result.x.reshape((m, n))
    total_cost = result.fun

    return allocation, total_cost

# genera le capacità dei centri, somma = factor * somma richieste
def generate_capacities(sumreq, numCenters, factor=1.1, power=1.5):
   """
   Generate integer capacities whose sum is approximately
   factor * sumreq and that grow superlinearly with the index.
   Parameters
      sumreq : int       Total demand.
      numCenters : int   Number of stores/centers.
      factor : float     Desired ratio total_capacity / total_demand.
      power : float (>1) Controls how fast capacities increase
   Returns
      capacities : list[int]
   """
   target = round(factor * sumreq)
   
   # Superlinear weights
   weights = np.array([(i + 1) ** power for i in range(numCenters)], dtype=float)
   
   # Scale to target total
   capacities = np.floor(target * weights / weights.sum()).astype(int)
   
   # Fix rounding error
   diff = target - capacities.sum()
   capacities[-1] += diff
   
   return capacities.tolist()

# calcolo costi h di magazzino
def solve_h(d, req):
   """
   Returns h such that:
     - h is nonincreasing as d increases
     - mean(h * req) == mean(d)
     - min(h) == min(d) / req[argmin(d)]
     - h is as close as possible (least squares) to alpha / d for some alpha
   """
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
   
   w = rs  # weights in mean(h * req)
   inv_d = 1.0 / ds
   
   lower = h_min * w.mean()
   if target_mean < lower * (1 - 1e-12):
      raise ValueError(
         f"Infeasible: mean(d) = {target_mean:.6g} is below "
         f"h_min * mean(req) = {lower:.6g}."
      )
   
   # Variables x = [h_0, ..., h_{n-1}, alpha]
   def objective(x):
      r = x[:n] - x[n] * inv_d
      return r @ r, np.r_[2 * r, -2 * (r @ inv_d)]
   
   # Equalities: mean(h*req) = mean(d), h[n-1] = h_min
   A_eq = np.zeros((2, n + 1))
   A_eq[0, :n] = w / n
   A_eq[1, n - 1] = 1.0
   b_eq = np.array([target_mean, h_min])
   
   # Monotonicity: h[i] - h[i+1] >= 0
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
      bounds=[(0.0, None)] * (n + 1),
      constraints=constraints,
      options={"ftol": 1e-12, "maxiter": 2000},
   )
   if not result.success:
      raise RuntimeError(f"Optimization failed: {result.message}")
   
   h = np.empty(n)
   h[order] = result.x[:n]
   return h
