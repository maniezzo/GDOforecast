'''
BESO-DRO: assume di avere già tutti i file dati calcolati da BESO-SAA
NOTA: quasi tutto il SW e' banalmente duplicato
'''
import pandas as pd, numpy as np
import json, random
from dromodel import go_dro
from generateArtificial import go_artificial

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
   np.random.seed(seed)
   random.seed(seed)
   
   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1, :].values
   print(f"Boost forecasting {name}")
   
   if(isUseCase):
      gs  = np.loadtxt(f"transpCosts{numscen}.csv", delimiter=",")
      pOA = oa_probabilities(gs)
      req = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
      cap = pd.read_csv("capacities.csv", header=None).values.flatten()
      # legge i costi
      dfcost = pd.read_csv("costs.csv")
      dist = np.array(dfcost.values)
      # legge le richieste di ogni scenario
      req = pd.read_csv(f"../boostset/req{numscen}.csv", header=None)
      go_dro(req,pOA,h,b,dist,cap,m,d,numscen, numCustomers, seed, test)
   else:
      boots,req,h,pOA,dist,cap = go_artificial(numCustomers,numCenters,numscen)
      go_dro(np.array(req).T,pOA,h,b,dist.T,cap,m,d,numscen, numCustomers, seed, test)

   print('Fine')

