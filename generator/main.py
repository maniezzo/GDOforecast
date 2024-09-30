import numpy as np
import pandas as pd
import json,random

# capacities and requests
def computeCapReq(cost):
   m = cost.shape[0]
   n = cost.shape[1]

   # richieste
   rq = np.zeros(n)
   for j in range(n):
      for i in range(m): rq[j] += cost.iloc[i,j]
   mx = max(rq)
   top = 1.1 * mx
   # a = top - a  # chi costa di più chiede di più

   # Final range
   min_new = 10
   max_new = 100
   min_rq = np.min(rq)
   max_rq = np.max(rq)

   # Rescale the array
   rescaled_array = min_new + (rq - min_rq) * (max_new - min_new) / (max_rq - min_rq)
   req = np.round(rescaled_array)

   # capacità
   cp = np.zeros(m)
   for i in range(m):
      for j in range(n): cp[i] += cost.iloc[i,j]
   mx = max(cp)
   top = 1.1 * mx

   # cost index of the servers
   sumreq = np.sum(req)
   min_new = 0.1
   max_new = 1
   min_cp = np.min(cp)
   max_cp = np.max(cp)

   # Rescale the array
   rescaled_array = min_new + (cp - min_cp) * (max_new - min_new) / (max_cp - min_cp)
   sumScaled = np.sum(rescaled_array)
   cap = np.round( 1.2 * (rescaled_array / sumScaled) * sumreq )

   sumcap = np.sum(cap)

   return cap,req

# cost for storing one unit of good. Assuming transportation and storage costs are compatible
def computeQcosts(cost):
   m = cost.shape[0]
   n = cost.shape[1]
   qcost = np.zeros(m)
   for i in range(m):
      sum = 0
      for j in range(n):
         sum += cost.iloc[i,j]/req[j]
      qcost[i] = np.round(sum/n)

   return qcost

if __name__ == "__main__":
   # row: server, col: client
   df = pd.read_csv('seedMatrix.csv',header=None, skiprows=1)

   # Randomly select a subset of rows and columns
   num_rows    = 50  # Number of rows to sample
   num_columns = 300  # Number of columns to sample
   num_mult    = 50   # number of allowed multiple assignments
   max_mult    = 5   # max num of multiple assignments

   isOrg = False # non servirà più
   if isOrg:
      name = "inst_52_4_0_0"  # i dati originali, veri
      dforg = df.iloc[0:52, 0:4].T
      dforg.to_csv("cost1.csv", header=False, index=False)
      req  = [27,23,28,23,19,17,19,14,28,17,20,15,27,26,29,23,21,14,18,12,28,30,34,18,17,20,17,19,11,28,11,14,16,29,12,18,29,25,20,26,14,9,17,26,14,17,20,5,14,21,32,16]
      cap  = [120,1000,300,180]
      qcost= [0,0,0,0]
      cols = np.arange(3)
      rows = np.arange(52)

   for k in range(5):
      # Randomly select rows
      instance_rows = df.T.sample(n=num_rows) #, random_state=42)  # Set random_state for reproducibility
      # Randomly select columns
      cost = instance_rows.sample(n=num_columns, axis=1) #, random_state=42)
      cap, req = computeCapReq(cost)
      qcost    = computeQcosts(cost)
      b = np.ones(num_columns).astype(int)
      id_to_change = random.sample(range(0, num_columns), num_mult)
      for id in id_to_change:
         b[id] = random.randint(1,max_mult+1)

      name = f"inst_{num_columns}_{num_rows}_{num_mult}_{k}"

      dict = {
               "name" : name,
               "n"    : len(cost.columns),
               "m"    : len(cost.index),
               "cols" : cost.columns.tolist(),
               "rows" : cost.index.tolist(),
               "qcost": qcost.tolist(),
               "req"  : req.tolist(),
               "cap"  : cap.tolist(),
               "b"    : b.tolist(),
               "nmult": num_mult
             }
      print(dict)

      with open(f'{name}.json', 'w') as file:
         json.dump(dict,file,indent=4)

   with open(f'{name}.json', 'r') as file:
      dct = json.load(file)

   name = dct["name"]
   n = dct["n"]
   m = dct["m"]
   cols = dct["cols"]
   rows = dct["rows"]
   req = dct["req"]
   cap = dct["cap"]
   b   = dct["b"]
   df2 = pd.read_csv('seedMatrix.csv',header=None, skiprows=1)
   df3 = df2.T.loc[rows,cols]
   df3 = df3.reset_index(drop=True)
   df3.columns = range(df3.shape[1])

   print("Fine")