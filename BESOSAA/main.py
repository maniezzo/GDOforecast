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

if __name__ == '__main__':
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   numscen = conf["numscen"]
   numCustomers = conf["numCustomers"]
   seed    = conf["seed"]
   b       = conf["b"]
   d       = conf["d"]
   gamma   = conf["gamma"]
   np.random.seed(seed)
   random.seed(seed)

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   print(f"Boost forecasting {name}")
   # inizio costruendomi il boostset (uguale a parità di seed)
   compute_boostset(df,numscen=numscen)

   req, gs = compute_transport(numscen)
   print("transportation costs:", gs)
   go_saa(req,gs,d,b,gamma,numscen,numCustomers)

   print('Fine')
