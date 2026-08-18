import numpy as np
import gurobipy as gp
from gurobipy import GRB

# scaling explicit otherwise high-volume stores can almost entirely determine the transportation metric
def wasserstein_scenario_distance_matrix( scenarios, scale=None, metric="l1"):
   """
   Pairwise ground distances between demand scenarios.
      scenarios : ndarray, shape (N, J)
      scale     : ndarray, shape (J,), or None
   Returns
      D : ndarray, shape (N, N)
      scale : scaling actually used
   """
   
   X = np.asarray(scenarios, dtype=float)
   N, J = X.shape
   
   if scale is None:
      # Simple dimensionless normalization.
      # You may instead use historical mean demand, std, MAD, etc.
      scale = np.mean(np.abs(X), axis=0)
      scale = np.maximum(scale, 1e-8)
   
   Z = X / scale
   
   if metric == "l1":
      D = np.abs( Z[:, None, :] - Z[None, :, :] ).mean(axis=2)
   elif metric == "l2":
      D = np.sqrt( ((Z[:, None, :] - Z[None, :, :]) ** 2).mean(axis=2) )
   else:
      raise ValueError("metric must be 'l1' or 'l2'")
   
   return D, scale

# choosing experimental radii parameterizes ε relative to scenario pairwise distance
def characteristic_distance(D):
   """
   Median non-zero off-diagonal pairwise distance.
   """
   D = np.asarray(D)
   vals = D[np.triu_indices_from(D, k=1)]
   vals = vals[vals > 0]
   
   if len(vals) == 0:
      return 1.0
   
   return np.median(vals)

# sensitività al raggio, calcolo raggio massimo ragionevole
def wasserstein_radius_diagnostics(D, D0):
   """
   Diagnostics for finite-support Wasserstein ambiguity set.

   D[s,t] = transport cost from empirical atom s to candidate atom t.
   """
   
   # Cost of moving the uniform empirical distribution
   # entirely onto each destination scenario t
   radius_to_pointmass = np.mean(D, axis=0)
   
   r_min = np.min(radius_to_pointmass)
   r_max = np.max(radius_to_pointmass)
   
   delta_min = r_min / D0
   delta_max = r_max / D0
   
   return  delta_max

# wassertein optimization objective
def add_wasserstein_objective(
    model,
    scenario_cost,
    D,
    epsilon,
    first_stage_cost=0.0,
    prefix="wd",
):
    """
    Add finite-support Wasserstein DRO objective.
    Parameters
      model    : gurobipy.Model
      scenario_cost : dict/list
                      scenario_cost[t] is a Gurobi linear expression representing
                      recourse cost under empirical demand scenario t.
      D        : ndarray (N, N) Wasserstein ground-distance matrix.
      epsilon  : float  Wasserstein radius.
      first_stage_cost : Gurobi expression  Cost independent of the uncertain scenario.
    Returns
      lam : Gurobi Var
      eta : Gurobi tupledict
    """

    D = np.asarray(D, dtype=float)
    N = D.shape[0]

    assert D.shape == (N, N)
    assert len(scenario_cost) == N

    # Wasserstein dual variable
    lam = model.addVar(
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name=f"{prefix}_lambda"
    )

    # Dual variables associated with empirical source atoms
    eta = model.addVars(
        range(N),
        lb=-GRB.INFINITY,
        vtype=GRB.CONTINUOUS,
        name=f"{prefix}_eta"
    )

    # ----------------------------------------------------------
    # eta_s >= C_t(x) - lambda * D_st
    #
    # for every empirical source atom s and possible destination t
    # ----------------------------------------------------------
    model.addConstrs(
        (
            eta[s]
            >= scenario_cost[t] - lam * float(D[s, t])
            for s in range(N)
            for t in range(N)
        ),
        name=f"{prefix}_dual"
    )

    wd_expected_worst_cost = (
        epsilon * lam
        + (1.0 / N) * gp.quicksum(
            eta[s] for s in range(N)
        )
    )

    model.setObjective(
        first_stage_cost + wd_expected_worst_cost,
        GRB.MINIMIZE
    )
    return lam, eta

# costruisce il modello determnistico di base, costo come da articolo
def build_base_model(p, rho, costs, d, Q, b):
   m, n = costs.shape
   nums, nr = rho.shape
   
   assert nr == n
   assert len(p) == nums
   assert len(Q) == m
   assert len(d) == m
   assert len(b) == n
   assert np.isclose(np.sum(p), 1.0)
   
   model = gp.Model("base_DE")
   
   I = range(m)
   J = range(n)
   S = range(nums)
   
   x = model.addMVar(
      (m, n),
      vtype=GRB.BINARY,
      name="x"
   )
   
   q = model.addMVar(
      (m, n, nums),
      lb=0.0,
      vtype=GRB.INTEGER,
      name="q"
   )
   
   eps = model.addMVar(
      (n, nums),
      lb=0.0,
      vtype=GRB.INTEGER,
      name="eps"
   )
   
   # Expected operating cost
   model.setObjective(
      gp.quicksum(
         costs[i, j] * x[i, j]
         for i in I for j in J
      )
      +
      gp.quicksum(
         p[s] * d[i] * q[i, j, s]
         for i in I for j in J for s in S
      )
      +
      gp.quicksum(
         p[s] * 1_000_000 * eps[j, s]
         for j in J for s in S
      ),
      GRB.MINIMIZE
   )
   
   model.addConstrs(
      (
         gp.quicksum(q[i, j, s] for i in I)
         + eps[j, s]
         == rho[s, j]
         for j in J for s in S
      ),
      name="demand"
   )
   
   model.addConstrs(
      (
         gp.quicksum(q[i, j, s] for j in J)
         <= Q[i]
         for i in I for s in S
      ),
      name="capacity"
   )
   
   model.addConstrs(
      (
         gp.quicksum(x[i, j] for i in I)
         == b[j]
         for j in J
      ),
      name="assignment"
   )
   
   model.addConstrs(
      (
         q[i, j, s] <= rho[s, j] * x[i, j]
         for i in I for j in J for s in S
      ),
      name="link"
   )
   
   return model, x, q, eps

# risolve MIP il modello di base
def solve_base_model(model):
   model.write("model.lp")  # human-readable LP format
   model.optimize()
   if model.Status == GRB.OPTIMAL:
      print(f"\nMIP Optimal objective : {model.ObjVal:.4f}")
      '''
      print("\n--- Primal variables ---")
      for i in I:
         for j in J:
            print(f" x{i}{j} = {x[i,j].X:.2f}")
      '''
   else:
      print(f"MIP did not solve to optimality. Status = {model.Status}")
   return

# controllo soluzione con dati storici
def checksol(x,cap,req):
   sumreq = np.zeros(len(cap))
   m,n = x.shape
   for i in range(m):
      for j in range(n):
         sumreq[i] += x[i,j]*req[j]
            
   for i in range(m):
      print(f"DC {i}: cap {cap[i]} sumreq {sumreq[i]}")
   return

# main della parte wassertein
def go_wasserstein(rho,D,D0,c,cap,delta_max):
   m,n = c.shape
   nums,_ = rho.shape
   I = range(m)
   J = range(n)
   S = range(nums)
   M = 1000000
   results = []
   # 7 valori di raggio fra 0 e eps*delta_max
   for eps_factor in np.linspace(0, delta_max, 7):
      epsilon = eps_factor * D0
      
      p = 1/nums * np.ones(len(rho))
      b = np.ones(n)
      d = np.zeros(m)
      model,x,q,slack = build_base_model(p, rho, c, d, cap, b)
      solve_base_model(model)
      
      first_stage_cost = gp.quicksum( c[i, j] * x[i, j] for i in I for j in J )
      
      scenario_cost = {
         s:
            gp.quicksum( d[i] * q[i,j,s] for i in I for j in J )
            +
            M * gp.quicksum( slack[j,s] for j in J )
         for s in S
      }
      
      lam, eta = add_wasserstein_objective(
         model,
         scenario_cost,
         D,
         epsilon,
         first_stage_cost
      )
      
      model.optimize()
      
      if model.SolCount > 0:
         results.append({
            "eps_factor": eps_factor,
            "epsilon": epsilon,
            "obj": model.ObjVal,
            "lambda": lam.X,
            "runtime": model.Runtime,
            "gap": model.MIPGap,
         })
      for j in J:
         for i in I:
            if(x[i,j].X > 0):
               print(f" ({i}_{j})")

      print(f"eps_factor: {eps_factor}, epsilon: {epsilon}, obj: {model.ObjVal}, lambda: {lam.X}, runtime: {model.Runtime}, gap: {model.MIPGap}")
      x_values = x.X
   return x_values
