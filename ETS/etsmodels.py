from ets_saa_scenarios import ets_saa_scenarios
import numpy as np, pandas as pd
import sys,os


# un po' di postprocessing sui risultati
def resultsKPI(cap, req, numscen, asgn, q, tol=1e-6):
   cap = np.asarray(cap, dtype=float)
   m = len(cap)
   loads = np.zeros((numscen, m))

   for i in range(m):
      for j in range(req.shape[1]):
         for s in range(numscen):
            loads[s, i] += q[i, j, s].X

   peak_loads = loads.max(axis=0)
   overloads = np.maximum(peak_loads - cap, 0.0)

   for i in range(m):
      print(
         f"DC {i}: capacity={cap[i]:.2f}, "
         f"peak delivery={peak_loads[i]:.2f}, "
         f"overload={overloads[i]:.6f}"
      )

   return loads, peak_loads, overloads, bool(np.all(overloads <= tol))

# ETS deterministico
def deterministic_ets_sap(
    ets_mean, c, d, Q, b, *,
    demand_scale=1.0,
    fixed_x=None,
):
    """
    Deterministic SAP using ETS point forecasts.
    q retains the scenario index: q[i, j, 0].
    No shortage is allowed.
    Input:
       ets_mean[j]: ETS forecast for the target month, BEFORE conversion, clipping and rounding; use info["mean"].
       demand_scale: Same conversion to two-week pallet requirements as in BESO.
       Requires the existing build_beso_saa function.
    Returns
      model, x, q, req
    """
    mu = np.asarray(ets_mean, dtype=float)

    if mu.ndim != 1 or mu.size == 0 or not np.isfinite(mu).all():
        raise ValueError("ets_mean must be a finite, nonempty vector.")

    scale = np.broadcast_to(
        np.asarray(demand_scale, dtype=float), mu.shape
    )
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("demand_scale must be finite and positive.")

    # Deterministic demand: convert first, then round up.
    req = np.ceil(np.maximum(mu, 0.0) * scale).astype(np.int64)

    #besosaa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../SAP'))
    #if besosaa_path not in sys.path:
    #   sys.path.append(besosaa_path)
    #from sapmodel import build_sap
    besosaa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../BESOSAA'))
    if besosaa_path not in sys.path:
       sys.path.append(besosaa_path)
    from saaModel import build_beso_saa
    model, x, q, eps = build_beso_saa(
        rho=req[None, :],       # shape (1, number_of_stores)
        p_OA=np.array([1.0]),
        c=c,
        d=d,
        Q=Q,
        b=b,
        M=1.0,                 # irrelevant: shortages fixed to zero
        fixed_x=fixed_x,
    )

    for var in eps.values():
        var.LB = 0.0
        var.UB = 0.0

    model.ModelName = "ETS_SAP"
    model.update()

    return model, x, q, req

def go_ets_saa(req, d, bind, numscen, numCustomers, seed):
   # history[t, j]: ORIGINAL monthly demand for store j.
   # For use case: shape (45, 52), excluding the test months.
   # conversion_factor: the same monthly-demand-to-two-week-pallet
   # conversion used in BESO; scalar or one factor per store.

   # legge i costi
   dfcost = pd.read_csv("costs.csv")
   cost = np.array(dfcost.values)
   # legge le capacita
   cap = pd.read_csv("capacities.csv", header=None).values.flatten()
   b = np.ones(numCustomers)
   for i in range(len(bind)): b[bind[i]] += 1

   '''
   Fits one ETS model per store.
   Extracts the predictive mean and standard deviation for month \(T+3\), not the sum of the next three months.
   Generates normal draws, without resampling historical observations or residuals.
   Converts demand to your optimization units, sets negative draws to zero, and rounds up.
   Returns rho[s, j] and uniform probabilities \(p_s=1/N\).
   '''
   print("Building ets SCENARIOS ...")
   rho, p, info = ets_saa_scenarios(
      req,
      n_scenarios=numscen,
      horizon=3,
      demand_scale=1,
      trend="add",
      damped_trend=True,
      seasonal_periods=12,   # annual seasonality in monthly data
      seed=seed,
   )

   besosaa_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../BESOSAA'))
   if besosaa_path not in sys.path:
      sys.path.append(besosaa_path)
   from saaModel import build_beso_saa

   # Same optimization function as before.
   model, x, q, eps = build_beso_saa(
      rho=rho,
      p_OA=p,
      c=cost,
      d=d,
      Q=cap,
      b=b,
      M=9999999,
   )
   model.optimize()

   print("Scenario matrix:", rho.shape)
   print("Forecast means:", info["mean"])
   print("Predictive standard deviations:", info["std"])

   if model.SolCount > 0:
      assignments = [(i, j) for i, j in x if x[i, j].X > 0.5]
      loads, peak_loads, overloads, indover = resultsKPI(cap,req,numscen,assignments,q)

      print(f"n.rows {model.NumConstrs} n.cols {model.NumVars} n.non0 {model.NumNZs}")
      print(f"Objective: {model.ObjVal:.3f} t.cpu {model.Runtime:.3f}")
      print("Peak loads:", peak_loads)
      print("Overloads:", overloads)
      #print(f"Assignments: {assignments}")
   else:
      print("No feasible solution found. Gurobi status:", model.Status)

   # modello deterministico,come SAA ma con un unico scenario
   print("---------------------------- ETS DETERMINISTICO")
   model, x, q, req_ets = deterministic_ets_sap(info["mean"], cost, d, cap, b)

   model.optimize()

   if model.SolCount > 0:
      print("ETS-SAP objective:", model.ObjVal)
      print("Deterministic demands:", req_ets)
      # Deliveries retain the third, singleton scenario index:
      # quantity = q[i, j, 0].X
      assignments = [(i, j) for i, j in x if x[i, j].X > 0.5]
      dc_loads = [
         sum(q[i,j,0].X for j in range(len(req)))
         for i in range(len(cap))
      ]

      print("Point demands:", req_ets)
      print(f"n.rows {model.NumConstrs} n.cols {model.NumVars} n.non0 {model.NumNZs}")
      print(f"Objective: {model.ObjVal:.3f} t.cpu {model.Runtime:.3f}")
      # print("Assignments:", assignments)
      print("DC loads:", dc_loads)
   else:
      print("No feasible solution found; status:", model.Status)
   return
