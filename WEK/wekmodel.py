import numpy as np, pandas as pd
import gurobipy as gp
from gurobipy import GRB


def build_wek(
    rho, c, d, Q, b,
    radius, M=9999999, fixed_x=None
):
    """
    Build the finite-support 1-Wasserstein DRO benchmark (WEK).
    Does not solve the model.

    Parameters
    ----------
    rho : array (N, n)
        Nonnegative demand scenarios rho[s, j].
        Each row is one joint demand scenario across all stores.
        The empirical baseline assigns probability 1/N to each row.
    c : array (m, n)
        Fixed assignment costs c[i, j].
    d : array (m,)
        Unit storage costs d[i].
    Q : array (m,)
        Fixed DC capacities, including the external DC.
    b : array (n,)
        Exact number of DC assignments per store.
    M : float
        Positive penalty per pallet of unmet demand.
    radius : float
        Nonnegative Wasserstein radius, using unscaled L1 distances.
    fixed_x : dict, optional
        Prescribed assignments {(i, j): 0 or 1}.
        All indices are zero-based.

    Returns
    -------
    model, x, q, eps, lam, eta, D
        x[i,j]   : common binary assignments.
        q[i,j,t] : continuous deliveries in scenario t.
        eps[j,t] : continuous shortages in scenario t.
        lam      : nonnegative Wasserstein dual variable.
        eta[s]   : dual envelope for baseline scenario s.
        D[s,t]   : L1 distance between demand scenarios s and t.
    """
    # ---------- Validate inputs ----------
    rho, c, d, Q, b = [
        np.asarray(v, dtype=float) for v in (rho, c, d, Q, b)
    ]

    if rho.ndim != 2 or min(rho.shape) == 0:
        raise ValueError("rho must have shape (N, n), with N,n > 0.")
    N, n = rho.shape

    if c.ndim != 2 or c.shape[1] != n or c.shape[0] == 0:
        raise ValueError("c must have shape (m, n), with m > 0.")
    m = c.shape[0]

    if d.shape != (m,) or Q.shape != (m,) or b.shape != (n,):
        raise ValueError("Inconsistent dimensions of d, Q, or b.")

    if not all(np.all(np.isfinite(v)) for v in (rho, c, d, Q, b)):
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

    # ---------- Pairwise L1 distances ----------
    # Row-by-row computation avoids an N x N x n temporary array.
    D = np.empty((N, N), dtype=float)
    for s in S:
        D[s, :] = np.abs(rho - rho[s]).sum(axis=1)

    # ---------- Create model and variables ----------
    model = gp.Model("WEK")

    x = model.addVars(I, J, vtype=GRB.BINARY, name="x")

    # Continuous recourse, as in your WEK formulation.
    q = model.addVars(I, J, S, lb=0, name="q")
    eps = model.addVars(J, S, lb=0, name="eps")

    lam = model.addVar(lb=0, name="lambda")
    eta = model.addVars(S, lb=-GRB.INFINITY, name="eta")

    # ---------- Feasibility constraints ----------
    model.addConstrs(
        (gp.quicksum(x[i, j] for i in I) == b[j]
         for j in J),
        name="assignments"
    )

    model.addConstrs(
        (gp.quicksum(q[i, j, t] for i in I) + eps[j, t]
         == rho[t, j]
         for j in J for t in S),
        name="demand"
    )

    model.addConstrs(
        (gp.quicksum(q[i, j, t] for j in J) <= Q[i]
         for i in I for t in S),
        name="capacity"
    )

    model.addConstrs(
        (q[i, j, t] <= rho[t, j] * x[i, j]
         for i in I for j in J for t in S),
        name="link"
    )

    if fixed_x is not None:
        for (i, j), value in fixed_x.items():
            x[i, j].LB = value
            x[i, j].UB = value

    # ---------- Wasserstein dual constraints ----------
    recourse = {
        t: (
            gp.quicksum(
                d[i] * q[i, j, t] for i in I for j in J
            )
            + M * gp.quicksum(eps[j, t] for j in J)
        )
        for t in S
    }

    model.addConstrs(
        (eta[s] >= recourse[t] - lam * D[s, t]
         for s in S for t in S),
        name="wasserstein"
    )

    # ---------- Objective ----------
    assignment_cost = gp.quicksum(
        c[i, j] * x[i, j] for i in I for j in J
    )

    model.setObjective(
        assignment_cost
        + radius * lam
        + gp.quicksum(eta[s] for s in S) / N,
        GRB.MINIMIZE
    )
    model.update()

    return model, x, q, eps, lam, eta, D

def go_wek(req, d, bind, numscen):
   '''
   d:    costi di handling per ogni DC
   bind: INDICI dei client assegnabili a 2 DC
   '''
   numCustomers = 52
   # legge i costi
   dfcost = pd.read_csv("costs.csv")
   cost = np.array(dfcost.values)
   # legge le capacita
   cap = pd.read_csv("capacities.csv", header=None).values.flatten()
   b = np.ones(numCustomers)
   for i in range(len(bind)): b[bind[i]] += 1
   # legge le richieste di ogni scenario
   # req = pd.read_csv(f"../boostset/req{numscen}.csv", header=None)
   wasserstein_radius = 1.0
   
   model, x, q, eps, lam, eta, D = build_wek(
      rho=req,
      c=cost,
      d=d,
      Q=cap,
      b=b,
      radius=wasserstein_radius
   )
   
   model.Params.MIPGap = 0.001
   model.optimize()
   
   if model.SolCount > 0:
      assignments = [(i, j) for i, j in x if x[i, j].X > 0.5]
      
      worst_recourse_bound = (
            wasserstein_radius * lam.X
            + sum(eta[s].X for s in eta) / len(eta)
      )
      
      print("Objective:", model.ObjVal)
      print("Worst expected recourse bound:", worst_recourse_bound)
      print("Assignments:", assignments)
   else:
      print("No feasible solution found. Gurobi status:", model.Status)
   return