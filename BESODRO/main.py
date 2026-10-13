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
   seed    = conf["seed"]
   b       = conf["b"]
   d       = conf["d"]
   np.random.seed(seed)
   random.seed(seed)

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   print(f"Boost forecasting {name}")
   
   gs = np.loadtxt(f"transpCosts{numscen}.csv", delimiter=",")
   req = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
   go_dro(req,gs,d,b,numscen)

   print('Fine')
