import numpy as np, pandas as pd, os
from util import prewhiten,recolor,ar_rolling_forecast

def read_TPdata(numscen):
   numCustomers = 52
   # legge i costi
   dfcost = pd.read_csv("costs.csv")
   cost = np.array(dfcost.values)
   # legge le capacita
   cap = pd.read_csv("capacities.csv", header=None).values.flatten()
   req = []
   for s in range(numscen):
      # calcola le richieste per ogni bootstrap (scenario)
      print(f"Scenario {s}")
      _req = []
      for i in range(numCustomers):
         # leggo serie bbotstapped e costruisco lo scenario
         df = pd.read_csv(f"../boostset/boost{i}.csv",header=None)
         x = df.iloc[s,:-3].to_numpy()

         # prewhitening, forecasting, recoloring and getting the 3mh point forecast
         pwx, params = prewhiten(x)  # prewhiten
         f = ar_rolling_forecast(pwx, 1, 3, "n")
         f48 = np.append(pwx, f)
         x_back = recolor(f48, params)  # recolr
         # print(f"serie {i} forecast {x_back[-3:]} <-> true {ds[-3:].to_numpy()}")
         r = np.ceil(x_back[-1])
         _req.append(r)
      req.append( np.array(_req))
   return cost, cap, np.array(req)

# solve transportation problem
from scipy.optimize import linprog
from scipy.sparse import kron, eye, csr_matrix

def compute_transport(numscen):
   if(os.path.exists(f"transpCosts{numscen}.csv")):
      gs = np.loadtxt(f"transpCosts{numscen}.csv", delimiter=",")
      reqGlob = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
      return reqGlob,gs

   cost, cap, reqGlob = read_TPdata(numscen)
   np.savetxt(f"../boostset/req{numscen}.csv", reqGlob, delimiter=",", fmt="%.0f")

   # vettore dei costi di trasporto per ogni scenario
   gs = []

   for s in range(numscen):
      """Min-cost transportation: servers (rows) with capacity cap,
      clients (columns) with request req as row in reqGlob. Returns flow matrix and total cost."""
      # cost, cap, req = map(np.asarray, (cost, cap, req))
      req = reqGlob[s]
      m, n = cost.shape
      if cap.sum() < req.sum():
         raise ValueError("infeasible: total capacity < total requests")

      ones_n = csr_matrix(np.ones((1, n)))
      ones_m = csr_matrix(np.ones((1, m)))
      A_ub = kron(eye(m), ones_n)  # each server: sum_j x_ij <= cap_i
      A_eq = kron(ones_m, eye(n))  # each client: sum_i x_ij == req_j

      res = linprog(cost.ravel(), A_ub=A_ub, b_ub=cap, A_eq=A_eq, b_eq=req,
                    bounds=(0, None), method="highs")
      # di interesse flussi: res.x.reshape(m, n), costo: res.fun
      if not res.success:
         raise RuntimeError(res.message)
      gs.append(res.fun)
   # salva costi e richieste su file csv
   with open(f"transpCosts{numscen}.csv", "w") as f:
      f.write(",".join(str(g) for g in gs))
   return reqGlob,np.array(gs)

