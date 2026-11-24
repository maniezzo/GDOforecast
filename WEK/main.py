'''
WEK:
NOTA: quasi tutto il SW e' banalmente duplicato
'''
import pandas as pd, numpy as np
import json, random
from wekmodel import go_wek
from generateArtificial_wek import generate_artificial

if __name__ == '__main__':
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   numscen = conf["numscen"]
   numCustomers = conf["numCustomers"]
   seed       = conf["seed"]
   isUseCase  = conf["isUseCase"]
   numCenters = conf["numCenters"]
   numRep     = conf["numRep"]
   maxt       = conf["maxt"] # max solver CPU time
   b = conf["b"]  # num centri che servono
   h = conf["h"]  # handling cost
   gamma = conf["gamma"]
   np.random.seed(seed)
   random.seed(seed)

   with open("../results.csv", "a") as f:
      f.write(f"ncust,{numCustomers},ncent,{numCenters},nscen,{numscen},"
              f"b,{b},gamma,{gamma},maxt,{maxt}--\n")

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1,:].values
   print(f"Non-boost forecasting {name}")

   for irep in range(numRep):
      if(isUseCase):
         history = np.array(df.iloc[:-3,:])
         # legge le capacita
         cap = pd.read_csv("capacities.csv", header=None).values.flatten()
         # legge i costi
         dfcost = pd.read_csv("costs.csv")
         cost = np.array(dfcost.values).T
         numCustomers = history.shape[1]
         numCenters   = len(cap)
         go_wek(history, cost, cap, h, b, numscen, numCustomers, seed, test, maxt)
      else:
         print("Non use case")
         history,req,h,p_OA,cost,cap,test = generate_artificial(numCustomers,numCenters,numscen)
         go_wek(history, cost, cap, h, b, numscen, numCustomers, seed, test, maxt)
      seed += 3

   print('Fine')
