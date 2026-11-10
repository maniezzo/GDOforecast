import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import json,random
from transportation import compute_transport
from MEBbootstrap import meboot
from saaModel import go_saa

def compute_boostset(df,numscen=125):
   for i in range(len(df.columns)):
      y = df.iloc[:,i].values
      rng = np.random.default_rng(seed)
      B = meboot(y, reps=numscen, trim=0.10, rng=rng)
      B.round(decimals=0, out=B)
      np.savetxt(f"../boostset/boost{i}.csv", B.T, delimiter=",", fmt="%.0f")

   return

# calcolo oa probabilities per lo use case reale
def oa_probabilities(g, gamma=1, a=None):
    """calcola objective-aware probabilities p^OA_s"""
    g = np.asarray(g, dtype=float)
    if g.ndim != 1 or g.size == 0 or not np.all(np.isfinite(g)):
        raise ValueError("g must be a nonempty vector of finite scores.")
    if not np.isfinite(gamma):
        raise ValueError("gamma must be finite.")

    if a is None:
        a = np.ones(g.size)  # Uniform baseline; normalization cancels
    else:
        a = np.asarray(a, dtype=float)
        if (a.shape != g.shape or not np.all(np.isfinite(a))
                or np.any(a < 0) or not np.any(a > 0)):
            raise ValueError("a must match g, be nonnegative, and have positive mass.")

    sd = g.std(ddof=0)
    g_scaled = (g - g.mean()) / sd if sd > 0 else np.zeros_like(g)

    # Compute in log space to avoid exponential overflow.
    positive = a > 0
    log_weights = np.log(a[positive]) + gamma * g_scaled[positive]
    weights = np.exp(log_weights - log_weights.max())

    p = np.zeros_like(g)
    p[positive] = weights / weights.sum()
    return p

# BESO-SAA
if __name__ == '__main__':
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   numscen   = conf["numscen"]
   numCustomers = conf["numCustomers"]
   numCenters   = conf["numCenters"]
   seed      = conf["seed"]
   isUseCase = conf["isUseCase"]
   numRep    = conf["numRep"]
   b = conf["b"]  # num centri che servono
   h = conf["h"]  # handling cost
   m = conf["m"]  # num distribuzioni
   d = conf["d"]  # num draws per distribuzione
   gamma   = conf["gamma"] # risk-avertness del decisore
   np.random.seed(seed)
   random.seed(seed)

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1,:].values
   print(f"Boost forecasting {name}")
   # inizio costruendomi il boostset (uguale a parità di seed)
   compute_boostset(df,numscen=numscen)

   if(isUseCase):
      req, gs = compute_transport(numscen)
      print("transportation costs:", gs)
      go_saa(req,gs,d,b,gamma,numscen,numCustomers, seed, test)
   else:
      boots,req,h,pOA,dist,cap = go_artificial(numCustomers,numCenters,numscen)

   print('Fine')
