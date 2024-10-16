import numpy as np, pandas as pd, time
import pulp

def makeModel(requests, costs, cap, b):
   ncol = 2*ncli*nser
   nx   = ncli*nser
   categ ='Binary'  # 'Continuous'
   X = pulp.LpVariable.dicts('X%s', (range(nx)),
                        cat=categ,
                        lowBound=0,
                        upBound=1)

   Q = pulp.LpVariable.dicts('Q%s', (range(nx,ncol)),
                        cat='Integer',
                        lowBound=0,
                        upBound=50)
   X.update(Q) # append Q to X

   # create the LP object and set up as a MINIMIZATION problem
   probl = pulp.LpProblem('GDO', pulp.LpMinimize)
   # cost function
   c = np.zeros(ncol)
   c[0:nx] = [costs[i,j] for i in np.arange(nser) for j in np.arange(ncli)]
   probl += sum(c[i] * X[i] for i in range(nx))

   # knapsack constraints Sum qij leq Qi
   nrows = 0
   for i in np.arange(nser):
      probl += sum(X[nx+i*ncli+j] for j in np.arange(0,ncli)) <= cap[i], "cap%d" % i
      nrows += 1

   # assignment constraints Sum qij = reqj
   for j in np.arange(ncli):
      probl += sum(X[nx+i*ncli+j] for i in np.arange(0,nser)) == requests[j], f"ass{j}"
      nrows += 1

   # amount constraint Sum xij leq bj
   for j in np.arange(ncli):
      probl += sum(X[i*ncli+j] for i in np.arange(0,nser)) <= b[j], f"b{j}"
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
   x = np.full(ncli,-1)
   for i in np.arange(ncol):
      v = probl.variables()[i]
      ind = int(v.name[1:])
      cli = ind%ncli
      ser = (ind//ncli)%nser # X and Q, repetition
      if(v.name[0]=='X' and v.varValue>0):
         x[cli]=ser
      sol.append(f"name {v.name} i {i} val {v.varValue}")
      #print(f"{v} = {v.varValue}  i: {cli}, ser: {ser}")
   #fout.write(str(sol))
   #fout.write(f"{np.array2string(x, max_line_width=10000,separator=',')}\r\n")
   return (cost,sol)

def subProblem(requests, costs, cap, b, vlambda):
   ncol = 2*ncli*nser
   nx   = ncli*nser
   categ ='Binary'  # 'Continuous'
   X = pulp.LpVariable.dicts('X%s', (range(nx)),
                        cat=categ,
                        lowBound=0,
                        upBound=1)

   Q = pulp.LpVariable.dicts('Q%s', (range(nx,ncol)),
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

   # x - q, force x
   for i in np.arange(nser):
      for j in np.arange(ncli):
         probl += X[nx+i*ncli+j] - requests[j]*X[i*ncli+j] <= 0, f"qx{nrows}"
         nrows += 1

   # xij <= qij, force q
   for i in np.arange(nser):
      for j in np.arange(ncli):
         probl += X[i*ncli+j] - X[nx+i*ncli+j] <= 0, f"xq{nrows}"
         nrows += 1

   # save the model in a lp file
   # probl.writeLP("GDOmodel.lp")
   # view the model
   # print(probl)

   # solve the model
   probl.solve(pulp.PULP_CBC_CMD(msg=0))
   cost = pulp.value(probl.objective) - add2
   #print(f"Subpr. status: {pulp.LpStatus[probl.status]} cost {cost}")
   # determinnig cost components
   qcost = 0
   sol = np.zeros(ncol)
   lstsol = []
   for i in np.arange(ncol):
      v = probl.variables()[i]
      if (v.varValue > 0):
         ii = int(v.name[1:])
         sol[ii]=v.varValue
         if ii>=nx:
            qcost += c[ii]
         lstsol.append({'cli': ii%ncli, 'ser': ii//ncli})
         #print(f"{v} = {v.varValue}  i: {ii}  cli {ii%ncli}, ser: {ii//ncli}")
   #print(f"Subpr. objective: {cost} qcost {qcost} add2 {add2}")
   #print(f"LR sol: {sol}")
   return (cost,sol)

def checkFeas(sol,cap, costs):
   isFeas = True
   subgrad = np.zeros(nser)
   nx = len(sol)//2
   # assignment constraints Sum qij = reqj
   for i in np.arange(nser):
      sum = 0
      for j in np.arange(ncli):
         sum += sol[nx + i * ncli + j]
      subgrad[i] = sum - cap[i]
      if sum > cap[i]:
         isFeas = False
   # check cost
   z = 0
   for i in np.arange(costs.size):
      ii = i // ncli
      jj = i % ncli
      z += sol[i]*costs[ii,jj]
   #print(f"Checked cost: {z}")

   return (isFeas, subgrad)

def computeZub(sol,costs,requests,cap):
   zub = 0
   freecap = np.array(cap)
   soliter = np.full(ncli,-1)
   indreq = np.argsort(requests)
   indreq = np.flip(indreq) # higher to smaller
   isFeasible = True
   for ii in np.random.permutation(ncli):
      fAssigned = False
      i = indreq[ii]
      server = int(0*sol[i]+1*sol[ncli+i]+2*sol[2*ncli+i]+3*sol[3*ncli+i])
      if(freecap[server] >= requests[i]):
         soliter[i] = server
         freecap[server] -= requests[i]
         zub += costs[server,i]
         fAssigned = True
      else:
         isFeasible = False
         srvCosts = costs[:,i]
         indc = np.argsort(srvCosts)
         for k in np.random.permutation(len(indc)):
            if (freecap[indc[k]] >= requests[i]):
               soliter[i] = indc[k]
               freecap[indc[k]] -= requests[i]
               zub += costs[indc[k], i]
               fAssigned = True
               break
      if not fAssigned:
         print(">>>>>>>>>>>>>>>>>>>>>>>> ASSIGNMENT ERROR <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

   if isFeasible:
      print(f"Feasible solution, cost {zub}")
   return isFeasible, soliter, zub

def subgradient(requests,costs,cap,b,alpha=0.1,niter=3):
   nuseless  = 0  # number of non improving iterations
   alphainit = alpha
   vlambda   = np.zeros(nser)
   iter = 0
   zub=50000
   zlb = 0
   while(iter < niter):
      print(f"SUBGR ===================== iter {iter}")
      (zliter,sol) = subProblem(requests,costs,cap,b,vlambda)
      (isFeas, subgrad) = checkFeas(sol,cap, costs)
      isFeasible, soliter, zubiter = computeZub(sol,costs,requests,cap)
      #fout.write(f"{np.array2string(soliter, max_line_width=10000, separator=',')}\n")
      if isFeasible != isFeas: print(">>>>>>>>>>>>>>>>> FEASIBILITY MISMATCH <<<<<<<<<<<<<<<<<<<")
      nuseless += 1
      if zliter > zlb:   # update lb
         zlb = zliter    # SHOULD SAVE THE BEST LB SOLUTION HERE
         nuseless = 0
      if zubiter < zub:  # update ub
         zub = zubiter   # SHOULD SAVE THE BEST UB SOLUTION HERE
         nuseless = 0
      if(isFeas):        # check for optimality
         isOpt = True
         for i in np.arange(len(subgrad)):
            if(vlambda[i]!=0 and subgrad[i]!=0):
               isOpt = False
         if isOpt:
            print(f"Trovato l'ottimo! zopt = zlb = {zlb}")
            return (zlb,sol)
      sub2 = 0           # not provably optimal
      for i in np.arange(nser): sub2 += subgrad[i]*subgrad[i]
      step = alpha*(zub - zlb)/sub2
      for i in np.arange(nser):
         vlambda[i] += step * subgrad[i]
         if(vlambda[i]<=0): vlambda[i]=0
      print(f"subgr, iter {iter} zlb= {zlb} zliter={zliter} zubiter={zubiter} zub={zub} step = {step}")
      #print(f"Lambda {vlambda}")
      #print(f"Subgr  {subgrad}")
      iter += 1
      if(iter%100 == 0):
         alpha = 0.8*alpha
         if alpha < 0.01:
            alpha = 0.01
      if nuseless > 100:
         nuseless = 0
         alpha = alphainit
         vlambda = 20*np.random.random(nser)

   return (zlb,sol)

if __name__ == "__main__":
   dfcosts = pd.read_csv("costs.csv")
   dfreq   = pd.read_csv("requests.csv")
   #fout = open("solutions.txt", "w")

   tstart = time.process_time()
   ncli = dfcosts.shape[1]
   nser = dfcosts.shape[0]
   b = np.ones(ncli)
   b[1] = 1

   fOptimal = True
   if fOptimal:
      (cost,sol) =  makeModel(dfreq.iloc[0,0:ncli].values,
                              dfcosts.iloc[0:nser,0:ncli].values,
                              dfreq.iloc[0:nser,ncli].values,
                              b)
      print(f"IP model, cost {cost}")

   (zLR,sol) =  subgradient(dfreq.iloc[0,0:ncli].values,
                           dfcosts.iloc[0:nser,0:ncli].values,
                           dfreq.iloc[0:nser,ncli].values,
                           b,alpha=4, niter = 1000)
   print(f"lagrangian model, cost {zLR}")

   tend = time.process_time()
   print("tcpu in seconds:", tend - tstart)
   #fout.close()
   pass