import numpy as np, pandas as pd
import gurobipy as gp
from gurobipy import GRB

def build_sap( forecasts, c, d, Q, b, fixed_x=None):
   """
   Parameters
      forecasts : array (N, n)
        Bootstrap demand forecasts on the original demand scale.
        Rows are bootstrap replicates; columns are stores.
        Prefer forecasts before individual ceiling rounding.
      c : array (m, n), Fixed assignment costs c[i, j].
      d : array (m,), Unit storage costs d[i].
      Q : array (m,), Fixed DC capacities, including the external DC.
      b : array (n,), Exact number of DC assignments per store.
      fixed_x : dict, Prescribed assignments {(i, j): 0 or 1}.
   Returns
      model, x, q, req
        x[i,j] : binary assignments.
        q[i,j] : integer deliveries.
        req[j] : ceiling of the mean bootstrap forecast for store j.
   """
   # Validazione input
   forecasts, c, d, Q, b = [
     np.asarray(v, dtype=float)
     for v in (forecasts, c, d, Q, b)
   ]

   if forecasts.ndim != 2 or min(forecasts.shape) == 0:
     raise ValueError("forecasts must have shape (N, n), with N,n > 0.")
   _, n = forecasts.shape

   if c.ndim != 2 or c.shape[1] != n or c.shape[0] == 0:
     raise ValueError("c must have shape (m, n), with m > 0.")
   m = c.shape[0]

   if d.shape != (m,) or Q.shape != (m,) or b.shape != (n,):
     raise ValueError("Inconsistent dimensions of d, Q, or b.")

   if not all( np.all(np.isfinite(v)) for v in (forecasts, c, d, Q, b)):
     raise ValueError("All input arrays must contain finite values.")

   if np.any(forecasts < 0) or np.any(Q < 0):
     raise ValueError("Forecasts and capacities must be nonnegative.")

   if np.any(b != np.floor(b)) or np.any((b < 1) | (b > m)):
     raise ValueError("b must contain integers between 1 and m.")

   I, J = range(m), range(n)

   if fixed_x is not None:
     for (i, j), value in fixed_x.items():
         if i not in I or j not in J or value not in (0, 1):
             raise ValueError("Invalid fixed assignment index or value.")

   # Bagged point forecasts
   # Average first, then round up.
   req = np.ceil(forecasts.mean(axis=0)).astype(np.int64)

   # Create model and variables
   model = gp.Model("SAP")

   x = model.addVars(I, J, vtype=GRB.BINARY, name="x")
   q = model.addVars(I, J, lb=0, vtype=GRB.INTEGER, name="q")

   # ----------------------------------------------- Constraint
   model.addConstrs(
     (gp.quicksum(q[i, j] for i in I) == int(req[j]) for j in J),
     name="demand"
   )

   model.addConstrs(
     (gp.quicksum(q[i, j] for j in J) <= Q[i] for i in I),
     name="capacity"
   )

   model.addConstrs(
     (gp.quicksum(x[i, j] for i in I) == b[j] for j in J),
     name="assignments"
   )

   model.addConstrs(
     (q[i, j] <= int(req[j]) * x[i, j] for i in I for j in J),
     name="link"
   )

   if fixed_x is not None:
     for (i, j), value in fixed_x.items():
         x[i, j].LB = value
         x[i, j].UB = value

    # ---------- Objective ----------
   model.setObjective(
        gp.quicksum( c[i, j] * x[i, j] + d[i] * q[i, j] for i in I for j in J ),
        GRB.MINIMIZE
   )
   model.update()
   return model, x, q, req
    
def go_sap(req,d,bind,numscen):
    '''
    d: costi di handling per ogni DC
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
    req = pd.read_csv(f"../boostset/req{numscen}.csv", header=None)

    model, x, q, req = build_sap(
                          forecasts=req,  # Shape (N, number_of_stores)
                          c=cost,
                          d=d,
                          Q=cap,
                          b=b
                         )

    model.Params.MIPGap = 0.001
    model.optimize()

    if model.SolCount > 0:
       assignments = [(i, j) for i, j in x if x[i, j].X > 0.5]
       dc_loads = [
             sum(q[i, j].X for j in range(len(req)))
             for i in range(len(cap))
       ]

       print("Point demands:", req)
       print(f"n.rows {model.NumConstrs} n.cols {model.NumVars} n.non0 {model.NumNZs}")
       print(f"Objective: {model.ObjVal:.3f} t.cpu {model.Runtime:.3f}")
       #print("Assignments:", assignments)
       print("DC loads:", dc_loads)
    else:
       print("No feasible solution found. Gurobi status:", model.Status)
    return