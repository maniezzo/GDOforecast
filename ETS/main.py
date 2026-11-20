import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import json,random
from etsmodels import go_ets_saa
from generateArtificial_ets import go_artificial

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
   b = conf["b"]  # num centri che servono
   h = conf["h"]  # handling cost
   np.random.seed(seed)
   random.seed(seed)

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1,:].values
   print(f"Non-boost forecasting {name}")

   if(isUseCase):
      req = np.array(df.iloc[:-3,:])
      go_ets_saa(req, h, b, numscen, numCustomers, seed, test)
   else:
      print("Non use case")
      _,req,h,p_OA,cost,cap,test = go_artificial(numCustomers,numCenters,numscen)
      go_ets_saa(req, h, b, numscen, numCustomers, seed, test)

   print('Fine')
