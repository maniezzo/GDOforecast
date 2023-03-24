import numpy as np, pandas as pd
import pulp

def makeModel(requests, costs, cap):
   ncol = 2*ncli*nser
   nx   = ncli*nser
   categ ='Binary'  # 'Continuous'
   X = pulp.LpVariable.dicts('X%s', (range(nx)),
                        cat=categ,
                        lowBound=0,
                        upBound=1)

   Y = pulp.LpVariable.dicts('Y%s', (range(nx,ncol)),
                        cat='Integer',
                        lowBound=0,
                        upBound=50)
   X.update(Y) # append Y to X

   # create the LP object, set up as a MINIMIZATION problem
   probl = pulp.LpProblem('GDO', pulp.LpMinimize)
   # cost function
   c = np.zeros(ncol)
   c[0:nx] = [costs[i,j] for i in np.arange(nser) for j in np.arange(ncli)]
   probl += sum(c[i] * X[i] for i in range(nx))

   # knapsack constraints
   nrows = 0
   for i in np.arange(nser):
      probl += sum(requests[i,j]*X[i*ncli+j] for j in np.arange(0,ncli)) <= cap[i], "cap%d" % nrows
      nrows += 1

   # assignment constraints
   for i in np.arange(ncli):
      probl += sum(X[j*ncli+i] for j in np.arange(0,nser)) == 1, f"ass{nrows}"
      nrows += 1

   # amount constraint
   for i in np.arange(nx):
      probl += sum(X[nx+j*ncli+i] for j in np.arange(0,nser)) >= 1, f"amt{nrows}"
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
   (cost,sol) =  makeModel(dfreq.iloc[0:nser,0:ncli].values,
                           dfcosts.iloc[0:nser,0:ncli].values,
                           dfreq.iloc[0:nser,ncli].values)

   pass