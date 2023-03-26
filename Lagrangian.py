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
   # probl.writeLP("GDOmodel.lp")
   # view the model
   # print(probl)

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
         #print(f"{v} = {v.varValue}  i: {i%ncli}, ser: {i//ncli}")
   return (cost,sol)

def subProblem(requests, costs, cap, b, vlambda):
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
   for i in np.arange(nser):
      for j in np.arange(ncli):
         c[nx+i*ncli+j] = vlambda[i]
   add2 = 0
   for i in np.arange(nser): add2 += vlambda[i]*cap[i]
   probl += sum(c[i] * X[i] for i in range(ncol))

   nrows = 0
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
   # probl.writeLP("GDOmodel.lp")
   # view the model
   # print(probl)

   # solve the model
   probl.solve(pulp.PULP_CBC_CMD(msg=0))
   cost = pulp.value(probl.objective)
   print(f"Subpr. status: {pulp.LpStatus[probl.status]} cost {cost}")
   print(f"add2 = {add2}")
   zlb = cost-add2
   print(f"Subpr. objective: {cost} zlb {zlb}")
   sol =np.zeros(ncol)
   for i in np.arange(ncol):
      v = probl.variables()[i]
      sol[i]=v.varValue
      #if (v.varValue > 0):
      #   sol.append({'i': i%ncli, 'ser': i//ncli})
         #print(f"{v} = {v.varValue}  i: {i%ncli}, ser: {i//ncli}")
   return (zlb,sol)

def checkFeas(sol,cap):
   isFeas = True
   subgrad = np.zeros(nser)
   nx = len(sol)//2
   # assignment constraints Sum qij = reqj
   for i in np.arange(nser):
      sum = 0
      for j in np.arange(ncli):
         sum += sol[nx + i * ncli + j]
      if sum > cap[i]:
         subgrad[i] = sum-cap[i]
         isFeas = False

   return (isFeas, subgrad)

def subgradient(requests,costs,cap,b):
   alpha = 0.001
   vlambda = np.zeros(nser)
   iter = 0
   zub=16000
   while(iter < 20):
      (zlb,sol) = subProblem(requests,costs,cap,b,vlambda)
      (isFeas, subgrad) = checkFeas(sol,cap)
      if(isFeas):
         print(f"Trovato l'ottimo! zopt = zlb = {zlb}")
         return
      else:
         print(f"subgr, iter {iter} zlb = {zlb}")
         sub2 = 0
         for i in np.arange(nser): sub2 += subgrad[i]
         step = (zub - zlb)/sub2
         for i in np.arange(nser):
            if(subgrad[i]==0): vlambda[i]=0
            else:
               vlambda[i] += step*subgrad[i]
      iter += 1

   return (zlb,sol)

if __name__ == "__main__":
   dfcosts = pd.read_csv("costs.csv")
   dfreq   = pd.read_csv("requests.csv")
   ncli = 52
   nser = 4
   b = np.ones(ncli)
   b[1] = 1
   (cost,sol) =  makeModel(dfreq.iloc[0,0:ncli].values,
                           dfcosts.iloc[0:nser,0:ncli].values,
                           dfreq.iloc[0:nser,ncli].values,
                           b)
   print(f"IP model, cost {cost}")

   (zLR,sol) =  subgradient(dfreq.iloc[0,0:ncli].values,
                           dfcosts.iloc[0:nser,0:ncli].values,
                           dfreq.iloc[0:nser,ncli].values,
                           b)
   print(f"lagrangian model, cost {zLR}")
   pass