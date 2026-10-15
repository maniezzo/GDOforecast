import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import json,random
from etsmodels import go_ets_saa


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
   print(f"Non-boost forecasting {name}")
   req = np.array(df.iloc[:-3,:])
   go_ets_saa(req, d, b, numscen, numCustomers, seed)

   print('Fine')
