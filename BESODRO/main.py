'''
BESO-DRO: assume di avere già tutti i file dati calcolati da BESO-SAA
NOTA: quasi tutto il SW e' banalmente duplicato
'''
import pandas as pd, numpy as np
import json, random
from dromodel import go_dro

if __name__ == '__main__':
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   numscen = conf["numscen"]
   numCustomers = conf["numCustomers"]
   seed    = conf["seed"]
   b = conf["b"]  # num centri che servono
   h = conf["h"]  # handling cost
   m = conf["m"]  # num distribuzioni
   d = conf["d"]  # num draws per distribuzione
   np.random.seed(seed)
   random.seed(seed)

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1,:].values
   print(f"Boost forecasting {name}")
   
   gs  = np.loadtxt(f"transpCosts{numscen}.csv", delimiter=",")
   req = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
   go_dro(req,gs,h,b,m,d,numscen, numCustomers, seed, test)

   print('Fine')
