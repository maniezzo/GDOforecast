'''
SAP: assume di avere già tutti i file dati calcolati da BESO-SAA
NOTA: quasi tutto il SW e' banalmente duplicato
'''
import pandas as pd, numpy as np
import json, random
from sapmodel import go_sap
'''
radius=0 ripristina l'SAA a peso uniforme, con gli stessi domini di recourse. Con domanda e capacità intere, questa transportation recourse ammette soluzioni ottimali intere nonostante la sua definizione continua.
Il radius utilizza la scala delle differenze totali assolute della domanda tra i negozi, non una percentuale. L'insieme di ambiguità si limita a ridistribuire la probabilità sui vettori di domanda forniti.
'''
if __name__ == '__main__':
   with open('config.json') as jconf:
      conf = json.load(jconf)
   print(conf)
   numscen = conf["numscen"]
   numCustomers = conf["numCustomers"]
   seed    = conf["seed"]
   b       = conf["b"]
   d       = conf["d"]
   np.random.seed(seed)
   random.seed(seed)

   name = "dataframe_nocovid_full"
   df = pd.read_csv(f"../data/{name}.csv", usecols=[i for i in range(1, 53)])
   test = df.iloc[-1,:].values
   print(f"Boost forecasting {name}")
   
   req = np.loadtxt(f"../boostset/req{numscen}.csv", delimiter=",")
   go_sap(req,d,b,numscen, numCustomers, seed, test)

   print('Fine')
