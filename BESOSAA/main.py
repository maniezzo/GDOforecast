import numpy as np, pandas as pd
import json,random
from transportation import compute_transport
from MEBbootstrap import meboot
from BESOsaaModel import go_besosaa, oa_probabilities, go_sap
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

# BESO-SAA
if __name__ == '__main__':
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   numscen   = conf["numscen"]
   numCustomers = conf["numCustomers"]
   numCenters   = conf["numCenters"]
   totdisp   = conf["totdisp"]
   seed      = conf["seed"]
   isUseCase = conf["isUseCase"]
   numRep    = conf["numRep"]
   maxt      = conf["maxt"] # max solver CPU time
   b = conf["b"]  # num centri che servono
   h = conf["h"]  # handling cost, for the use case
   m = conf["m"]  # num distribuzioni
   d = conf["d"]  # num draws per distribuzione
   gamma   = conf["gamma"] # risk-avertness del decisore
   np.random.seed(seed)
   random.seed(seed)

   with open("../results.csv", "a") as f:
      f.write(f"ncust,{numCustomers},ncent,{numCenters},nscen,{numscen},"
              f"b,{b},gamma,{gamma},totdisp,{totdisp},maxt,{maxt}--\n")

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1,:].values
   print(f"Boost forecasting {name}")

   for irep in range(numRep):
      # inizio costruendomi il boostset (su disco, uguale a parità di seed)
      compute_boostset(df,numscen=numscen)
   
      if(isUseCase):
         req, gs = compute_transport(numscen)
         print("transportation costs:", gs)
         p_OA = oa_probabilities(gs,gamma)
         req = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
         # legge le capacita
         cap = pd.read_csv("capacities.csv", header=None).values.flatten()
         # legge i costi
         dfcost = pd.read_csv("costs.csv")
         cost = np.array(dfcost.values)
         numCustomers = req.shape[1]
         numCenters   = len(cap)
         go_besosaa(req, cap, p_OA, h, b, cost, numscen, numCustomers, seed, test, maxt)
         go_sap(req, cap, h, b, cost, test, numCustomers=numCustomers, maxt=maxt)
      else:
         print("Non use case")
         boots,history,h,p_OA,cost,cap,testBoost,test = go_artificial(numCustomers,numCenters,numscen,seed,totdisp=totdisp)
         true_cost = go_besosaa(np.array(history).T, cap, p_OA, h, b, cost.T, numscen, numCustomers, seed, test, maxt)
         print(f"True cost={true_cost}")
         fDeterninistic = False
         if(fDeterninistic):
            go_sap(np.array(history).T, cap, h, b, cost.T, test, numCustomers=numCustomers, maxt=maxt)
      seed += 3

   print('Fine')
