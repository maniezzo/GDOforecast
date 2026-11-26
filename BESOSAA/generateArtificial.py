import numpy as np, pandas as pd
from MEBbootstrap import meboot
from statsmodels.tsa.ar_model import AutoReg
from scipy.optimize import linprog
from scipy.optimize import minimize, LinearConstraint
from prewhiten_recolor import prewhiten,recolor

def go_artificial(numCustomers,numCenters,numscen,gamma=1,totdisp=1.1):
   """
   totdisp: somma delle disponibilità dei DC rispetto a richiesta totale
   """
   name = "dataframe_nocovid_full"
   # train series
   dfseries = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   df2, trueVal = add_bootstrap_columns(dfseries, n_new_cols=numCustomers, random_state=995)
   print(df2.head())

   # distanze: prime righe/colunne centri, poi clienti
   dist = pd.read_csv(f"../matrice0.csv",header=None).values
   dist = dist[numCenters:(numCenters+numCustomers),:numCenters]
   
   # bootstrapping
   boots=[] # lista (3d) serie bootstrappate
   fcast=[] # lista (2d) previsioni serie in boots
   req = [] # bagged bootstrap forecasts
   testValBoost = [] # valori reali per ciascuna delle serie bootstrappate
   testVal = [] # valori reali per ciascuna serie selezionata
   for i in range(numCustomers):
      print(f"Bootsrapping client {i}")
      b = meboot(df2.iloc[:,i],reps=numscen) # MEB bootstrap
      f = [] # una previsione per ogni serie bootstrappata
      testVal.append(trueVal[i])
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
         testValBoost.append(trueVal[i])
         
      req[i] = np.ceil(req[i]/numscen) # previsione assoluta bootstrap
      boots.append(b)                  # serie bootstrappate
      fcast.append(np.array(f))        # previsioni serie bootstrap
      
   print("Defining servers' capacities and costs ...")
   # definizione capacità center note le richieste
   cap = generate_capacities(sum(req), numCenters, factor=totdisp, power=1.1)
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
   return boots,fcast,h,pOA,dist,cap,testValBoost,testVal

# estrae con replacement n_new_cols serie da df, per creare istanze artificiali
def add_bootstrap_columns(df, n_new_cols, random_state=None):
   """
   Parameters
      df : pd.DataFrame     DataFrame with exactly 48 numerical rows.
      n_new_cols : int      Number of new columns to generate.
      random_state : int or None Seed for reproducibility.
   Returns
   new_df : pd.DataFrame
       DataFrame with n_new_cols columns (named boot0, boot1, ...),
       each containing the first 45 values of the sampled original column.
   trueval : np.ndarray
       Array of length n_new_cols containing the 48th value (row index 47)
       of each sampled original column.
   """
   rng = np.random.default_rng(random_state)

   # Sample column names with replacement
   sampled_cols = rng.choice(df.columns, size=n_new_cols, replace=True)

   # Select the sampled columns (duplicates allowed)
   sampled_df = df[sampled_cols]

   # Extract the 48th value (index 47) of each sampled column
   trueval = sampled_df.iloc[47].values

   # Keep only the first 45 rows (indices 0..44)
   new_df = sampled_df.iloc[:45].copy()

   # Rename the columns to boot0, boot1, ...
   new_df.columns = [f"boot{i}" for i in range(n_new_cols)]

   return new_df, trueval

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
