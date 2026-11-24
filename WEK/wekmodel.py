from ets_saa_scenarios import ets_saa_scenarios
import numpy as np, pandas as pd
import gurobipy as gp
from gurobipy import GRB
import sys,os,time

def build_wek(
    rho, c, h, Q, b,
    radius, M=9999999, fixed_x=None
):
    """
    Build the finite-support 1-Wasserstein DRO benchmark (WEK).
    Parameters
       rho : array (N, n)
           Nonnegative demand scenarios rho[s, j].
           Each row is one joint demand scenario across all stores.
           The empirical baseline assigns probability 1/N to each row.
       c : array (m, n) Fixed assignment costs c[i, j].
       h : array (m,) Unit storage costs h[i].
       Q : array (m,) Fixed DC capacities, including the external DC.
       b : array (n,) Exact number of DC assignments per store.
       M : floatPositive penalty per pallet of unmet demand.
       radius : float Nonnegative Wasserstein radius, using unscaled L1 distances.
       fixed_x : dict, optional Prescribed assignments {(i, j): 0 or 1}.
    Returns
       model, x, q, eps, lam, eta, D
           x[i,j]   : common binary assignments.
           q[i,j,t] : continuous deliveries in scenario t.
           eps[j,t] : continuous shortages in scenario t.
           lam      : nonnegative Wasserstein dual variable.
           eta[s]   : dual envelope for baseline scenario s.
           D[s,t]   : L1 distance between demand scenarios s and t.
    """
    # Validazione input
    rho, c, h, Q, b = [ np.asarray(v, dtype=float) for v in (rho, c, h, Q, b) ]

    if rho.ndim != 2 or min(rho.shape) == 0:
        raise ValueError("rho must have shape (N, n), with N,n > 0.")
    N, n = rho.shape

    if c.ndim != 2 or c.shape[1] != n or c.shape[0] == 0:
        raise ValueError("c must have shape (m, n), with m > 0.")
    m = c.shape[0]

    if h.shape != (m,) or Q.shape != (m,) or b.shape != (n,):
        raise ValueError("Inconsistent dimensions of h, Q, or b.")

    if not all(np.all(np.isfinite(v)) for v in (rho, c, h, Q, b)):
        raise ValueError("All input arrays must contain finite values.")

    if np.any(rho < 0) or np.any(Q < 0):
        raise ValueError("Demands and capacities must be nonnegative.")

    if np.any(b != np.floor(b)) or np.any((b < 1) | (b > m)):
        raise ValueError("b must contain integers between 1 and m.")

    if not np.isfinite(M) or M <= 0:
        raise ValueError("M must be finite and positive.")

    if not np.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative.")

    I, J, S = range(m), range(n), range(N)

    if fixed_x is not None:
        for (i, j), value in fixed_x.items():
            if i not in I or j not in J or value not in (0, 1):
                raise ValueError("Invalid fixed assignment index or value.")

    # Pairwise L1 distances
    D = np.empty((N, N), dtype=float)
    for s in S:
        D[s, :] = np.abs(rho - rho[s]).sum(axis=1)

    # Create model and variables
    model = gp.Model("WEK")

    x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

    # Continuous recourse, see WEK formulation.
    q = model.addVars(I, J, S, lb=0, name="q")
    eps = model.addVars(J, S, lb=0, name="eps")

    lam = model.addVar(lb=0, name="lambda")
    eta = model.addVars(S, lb=-GRB.INFINITY, name="eta")

    # ---------------------------------------------------- constraints
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in I) == b[j] for j in J),
        name="assignments"
    )

    model.addConstrs(
        (gp.quicksum(q[i, j, t] for i in I) + eps[j, t] == rho[t, j] for j in J for t in S),
        name="demand"
    )

    model.addConstrs(
        (gp.quicksum(q[i, j, t] for j in J) <= Q[i] for i in I for t in S),
        name="capacity"
    )

    model.addConstrs(
        (q[i, j, t] <= rho[t, j] * x[i, j] for i in I for j in J for t in S),
        name="link"
    )

    if fixed_x is not None:
        for (i, j), value in fixed_x.items():
            x[i, j].LB = value
            x[i, j].UB = value

    # ----------------------------------- Wasserstein dual constraints

    recourse = {
        t: ( gp.quicksum( h[i] * q[i, j, t] for i in I for j in J )
                          + M * gp.quicksum(eps[j, t] for j in J)
           ) for t in S
    }
    model.addConstrs(
        (eta[s] >= recourse[t] - lam * D[s, t] for s in S for t in S),
        name="wasserstein"
    )

    # ------------------------------------ Objective
    assignment_cost = gp.quicksum( c[i, j] * x[i, j] for i in I for j in J )

    model.setObjective( assignment_cost + radius * lam+ gp.quicksum(eta[s] for s in S) / N,
                        GRB.MINIMIZE
                      )
    model.update()
    return model, x, q, eps, lam, eta, D

# controlla overload sui dati veri
def check_test(assgn,cap,test):
   m = len(cap)
   loads = np.zeros(m)

   for (dc,j) in assgn:
      loads[dc] += test[j]

   overload = np.zeros(m)
   for i in range(m):
      if(loads[i] >= cap[i]): overload[i] = loads[i]-cap[i]
   return loads, overload

# un po' di postprocessing sui risultati
def resultsKPI(name,cap, req, numscen, asgn, q, cost, h, trueVal, tol=1e-6):
    cap = np.asarray(cap, dtype=float)
    cost = np.asarray(cost, dtype=float)
    h = np.asarray(h, dtype=float)
    trueVal = np.asarray(trueVal, dtype=float)

    m = len(cap)
    n = req.shape[1]
    if len(trueVal) != n:
        raise ValueError("trueVal must contain one request per customer")

    # Loads in the model's scenarios
    loads = np.zeros((numscen, m))
    for i in range(m):
        for j in range(n):
            for s in range(numscen):
                loads[s,i] += q[i,j,s].X

    peak_loads = loads.max(axis=0)
    overloads = np.maximum(peak_loads - cap, 0.0)

    for i in range(m):
        print(
            f"DC {i}: capacity={cap[i]:.2f}, "
            f"peak delivery={peak_loads[i]:.2f}, "
            f"overload={overloads[i]:.6f}"
        )

    # Cost and loads for the requests that actually occurred
    assignment_cost = sum(cost[i, j] for i, j in asgn)
    true_loads = np.zeros(m)
    for i, j in asgn:
        true_loads[i] += trueVal[j]

    handling_cost = np.dot(h, true_loads)
    true_cost = assignment_cost + handling_cost
    true_overloads = np.maximum(true_loads - cap, 0.0)

    print(
        f"Realized cost: assignment={assignment_cost:.2f}, "
        f"handling={handling_cost:.2f}, total={true_cost:.2f}"
    )
    for i in range(m):
        print(
            f"DC {i}: realized delivery={true_loads[i]:.2f}, "
            f"realized overload={true_overloads[i]:.6f}"
        )
    return (
        loads, peak_loads, overloads,
        bool(np.all(overloads <= tol)),
        true_loads, true_overloads,
        assignment_cost, handling_cost, true_cost
    )

def go_wek(history, cost, cap, h, bind, numscen, numCustomers, seed, test, maxt):
   '''
   history[t, j]: ORIGINAL monthly demand for store j.
   h:    costi di handling per ogni DC
   bind: INDICI dei client assegnabili a 2 DC
   '''
   b = np.ones(numCustomers)
   for i in range(len(bind)): b[bind[i]] += 1

   # these are just shorthands
   n  = numCustomers
   m  = len(cap)
   ns = numscen

   '''
   Fits one ETS model per store.
   Extracts the predictive mean and standard deviation for month \(T+3\), not the sum of the next three months.
   Generates normal draws, without resampling historical observations or residuals.
   Converts demand to your optimization units, sets negative draws to zero, and rounds up.
   Returns rho[s, j] and uniform probabilities \(p_s=1/N\).
   '''
   tstart = time.process_time()
   print("Building ets SCENARIOS ...")
   rho, p, info = ets_saa_scenarios(
      history,
      n_scenarios=numscen,
      horizon=3,
      demand_scale=1,
      trend="add",
      damped_trend=True,
      seasonal_periods=12,   # annual seasonality in monthly data
      seed=seed,
   )
   tscen = time.process_time() - tstart # time for scenario preparation

   wasserstein_radius = 1.0
   model, x, q, eps, lam, eta, D = build_wek(
      rho=rho,
      c=cost.T,
      h=h,
      Q=cap,
      b=b,
      radius=wasserstein_radius
   )

   # --- Bias strongly toward finding feasible solutions early ---
   model.Params.MIPFocus = 1  # Primary feasibility focus
   model.Params.NoRelHeurTime = 10  # Up to 10 sec for pre-root NoRel heuristic
   # model.Params.Heuristics = 0.5  # Spend 50% of effort on primal heuristics

   # --- Strengthen neighborhood‑search heuristics ---
   model.Params.SubMIPNodes = 2000  # Let RINS explore larger neighborhoods
   model.Params.PumpPasses = 10  # More feasibility‑pump passes

   model.Params.TimeLimit = maxt  # or model.setParam('TimeLimit', 900), in seconds
   model.Params.MIPGap = 0.0001  # 0.01% relative optimality gap
   model.optimize()
   twek = time.process_time() - tscen
   
   if model.SolCount > 0:
      status = "opt" if model.Status==2 else "heu"
      assignments = [(i, j) for i, j in x if x[i, j].X > 0.5]
      worst_recourse_bound = ( wasserstein_radius * lam.X+ sum(eta[s].X for s in eta) / len(eta) )
      _, peak_loads, overloads, indover, true_loads, true_overloads, assignment_cost, handling_cost, true_cost = resultsKPI("WEK",cap,rho,numscen,assignments,q,cost.T,h,test)
      with open("../results.csv", "a") as f:
         f.write(f"WEK,n,{n},m,{m},ns,{ns}, status {status}, tscen,{tscen:.2f},tetssaa,{twek:.2f},"
                 f"assgn,{assignment_cost:.2f},"
                 f"hndl,{handling_cost:.2f},"
                 f"tot,{true_cost:.2f},"
                 f"overloads,{overloads},"
                 f"true overloads,{true_overloads}\n")
      loads,overloads = check_test(assignments,cap,test)

      print(f"n.rows {model.NumConstrs} n.cols {model.NumVars} n.non0 {model.NumNZs}")
      print(f"Objective: {model.ObjVal:.3f} t.cpu {model.Runtime:.3f}")
      print("Worst expected recourse bound:", worst_recourse_bound)
      #print("Assignments:", assignments)
      print("Peak loads:", peak_loads)
      print(f"True loads {loads} Overloads:, {overloads}")
   else:
      print("No feasible solution found. Gurobi status:", model.Status)
   return