import numpy as np, pandas as pd
import pulp

def makeModel(requests, costs, cap, b):
   ncol = 2*ncli*nser
   nx   = ncli*nser
   categ ='Binary'  # 'Continuous'
   X = pulp.LpVariable.dicts('X%s', (range(nx)),
                        cat=categ,
                        lowBound=0,
                        upBound=1)

   Q = pulp.LpVariable.dicts('Y%s', (range(nx,ncol)),
                        cat='Integer',
                        lowBound=0,
                        upBound=50)
   X.update(Q) # append Q to X

   # create the LP object, set up as a MINIMIZATION problem
   probl = pulp.LpProblem('GDO', pulp.LpMinimize)
   # cost function
   c = np.zeros(ncol)
   c[0:nx] = [costs[i,j] for i in np.arange(nser) for j in np.arange(ncli)]
   probl += sum(c[i] * X[i] for i in range(nx))

   # knapsack constraints Sum qij leq Qi
   nrows = 0
   for i in np.arange(nser):
      probl += sum(X[nx+i*ncli+j] for j in np.arange(0,ncli)) <= cap[i], "cap%d" % nrows
      nrows += 1

   # assignment constraints Sum qij = reqj
   for j in np.arange(ncli):
      probl += sum(X[nx+i*ncli+j] for i in np.arange(0,nser)) == requests[j], f"ass{nrows}"
      nrows += 1

   # amount constraint Sum xij leq bj
   for j in np.arange(ncli):
      probl += sum(X[i*ncli+j] for i in np.arange(0,nser)) <= b[j], f"b{nrows}"
      nrows += 1

   # x - q
   for i in np.arange(nser):
      for j in np.arange(ncli):
         probl += X[nx+i*ncli+j] - requests[j]*X[i*ncli+j] <= 0, f"xq{nrows}"
         nrows += 1

   # save the model in a lp file
   probl.writeLP("GDOmodel.lp")
   # view the model
   print(probl)

   # solve the model
   probl.solve()
   print("Status:", pulp.LpStatus[probl.status])
   cost = pulp.value(probl.objective)
   print("Objective: ", cost)
   sol = []
   for i in np.arange(ncol):
      v = probl.variables()[i]
      if (v.varValue > 0):
         sol.append({'i': i%ncli, 'ser': i//ncli})
         print(f"{v} = {v.varValue}  i: {i%ncli}, ser: {i//ncli}")
   return (cost,sol)

if __name__ == "__main__":
   dfcosts=pd.read_csv("costs.csv")
   dfreq  =pd.read_csv("requests.csv")
   ncli = 52
   nser = 4
   b = np.ones(ncli)
   b[1] = 1
   (cost,sol) =  makeModel(dfreq.iloc[0,0:ncli].values,
                           dfcosts.iloc[0:nser,0:ncli].values,
                           dfreq.iloc[0:nser,ncli].values,
                           b)

   pass