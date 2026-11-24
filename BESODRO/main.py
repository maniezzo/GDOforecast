import pandas as pd, numpy as np
import json, random
from transportation import compute_transport
from MEBbootstrap import meboot
from BESOdroModel import go_BESOdro, oa_probabilities
from generateArtificial import go_artificial

# definizione del bootset via MEB bootstrapping
def compute_boostset(df,numscen=125):
   for i in range(len(df.columns)):
      y = df.iloc[:,i].values
      rng = np.random.default_rng(seed)
      B = meboot(y, reps=numscen, trim=0.10, rng=rng)
      B.round(decimals=0, out=B)
      np.savetxt(f"../boostset/boost{i}.csv", B.T, delimiter=",", fmt="%.0f")
   return

# BESO-DRO
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
   maxt      = conf["maxt"] # max solver CPU time
   b = conf["b"]  # num centri che servono
   h = conf["h"]  # handling cost
   m = conf["m"]  # num distribuzioni
   d = conf["d"]  # num draws per distribuzione
   gamma   = conf["gamma"] # risk-avertness del decisore
   np.random.seed(seed)
   random.seed(seed)

   with open("../results.csv", "a") as f:
      f.write(f"ncust,{numCustomers},ncent,{numCenters},nscen,{numscen},"
              f"b,{b},gamma,{gamma},maxt,{maxt}--\n")

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1, :].values
   print(f"Boost forecasting {name}")
   
   for irep in range(numRep):
      # inizio costruendomi il boostset (su disco, uguale a parità di seed)
      compute_boostset(df, numscen=numscen)
   
      if(isUseCase):
         req, gs = compute_transport(numscen)
         print("transportation costs:", gs)
         pOA = oa_probabilities(gs,gamma)
         req = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
         # legge le capacita
         cap = pd.read_csv("capacities.csv", header=None).values.flatten()
         # legge i costi
         dfcost = pd.read_csv("costs.csv")
         dist = np.array(dfcost.values)
         # legge le richieste di ogni scenario
         numCustomers = req.shape[1]
         numCenters   = len(cap)
         go_BESOdro(req, pOA, h, b, dist, cap, m, d, numscen, numCustomers, seed, test, maxt)
      else:
         print("Non use case")
         boots,history,h,pOA,cost,cap,testBoost,test = go_artificial(numCustomers,numCenters,numscen,seed)
         go_BESOdro(np.array(history).T, pOA, h, b, cost.T, cap, m, d, numscen, numCustomers, seed, test, maxt)
      seed += 3

   print('Fine')

