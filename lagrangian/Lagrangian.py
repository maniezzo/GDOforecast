import numpy as np, pandas as pd, time
import pulp
import LocSearch as LS
import json, time, random

def makeMIPmodel(costs, qcost, requests, cap, b, isInteger=True):
   solver_list = pulp.listSolvers(onlyAvailable=True)
   print(solver_list)
   if "CPLEX_CMD" in solver_list:
         solver = pulp.getSolver('CPLEX_CMD',timeLimit=600,msg=False)
   else: solver = pulp.getSolver('PULP_CBC_CMD')
   ncol = 2*ncli*nser
   nx   = ncli*nser
   if(isInteger):
      categX = 'Binary' # 'Binary'  'Continuous'
      categQ = 'Integer' # 'Integer'  'Continuous'
   else:
      categX = 'Continuous' # 'Binary'  'Continuous'
      categQ = 'Continuous' # 'Integer'  'Continuous'
   X = pulp.LpVariable.dicts('X%s', (range(nx)),
                        cat=categX,
                        lowBound=0,
                        upBound=1)

   Q = pulp.LpVariable.dicts('Q%s', (range(nx,ncol)),
                        cat=categQ,
                        lowBound=0,
                        upBound=max(requests))
   X.update(Q) # append Q to X

   # create the LP object and set up as a MINIMIZATION problem
   probl = pulp.LpProblem('GDO', pulp.LpMinimize)
   # cost function
   c = np.zeros(ncol)
   c[0:nx] = [costs[i,j] for i in np.arange(nser) for j in np.arange(ncli)]
   c[nx:ncol] = [qcost[i] for i in np.arange(nser) for j in np.arange(ncli)]
   probl += sum(c[i] * X[i] for i in range(ncol))

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
   probl.solve(solver=solver)
   print("Status:", pulp.LpStatus[probl.status])
   cost = pulp.value(probl.objective)
   print("Objective: ", cost)
   sol = []
   if(ncol<1000):
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

def checkFeas(sol,cap,req,costs):
   isFeas = True
   subgradCap = np.zeros(nser)
   nx = len(sol)//2
   # capacity constraints Sum qij <= cap i
   for i in np.arange(nser):
      sum = 0
      for j in np.arange(ncli):
         sum += sol[nx + i * ncli + j]
      subgradCap[i] = sum - cap[i]
      if sum > cap[i]:
         isFeas = False
   # requests constraints Sum qij = reqj
   for j in np.arange(ncli):
      sum = 0
      for i in np.arange(nser):
         sum += sol[nx + i * ncli + j]
      if sum != req[j]:
         isFeas = False
   # assignment constraints Sum xij = bi
   for i in np.arange(nser):
      sum = 0
      for j in np.arange(ncli):
         sum += sol[i * ncli + j]
      if sum != b[i]:
         isFeas = False
   # linking constraints
   subgradLink = np.zeros(nser*ncli)
   for i in np.arange(nser):
      for j in np.arange(ncli):
         subgradLink[i*ncli + j] = sol[nx + i * ncli + j] - req[j]*sol[i*ncli + j]
         if(sol[nx + i*ncli + j] > req[j]*sol[i*ncli + j]):
            isFeas = False
   # check cost
   z = np.infty
   if(isFeas):
      z = 0
      for i in np.arange(costs.size):
         ii = i // ncli
         jj = i % ncli
         z += sol[i]*costs[ii,jj]
      #print(f"Checked cost: {z}")

   return (isFeas, subgradCap, subgradLink, z)

def computeFOval(sol, costs, requests, cap):
   z = 0
   freecap = np.array(cap)
   soliter = np.full(ncli,-1)
   indreq = np.argsort(requests)
   indreq = np.flip(indreq) # higher to smaller
   isFeasible = True
   for ii in np.random.permutation(ncli):
      fAssigned = False
      i = indreq[ii]
      server = int(0*sol[i]+1*sol[ncli+i]+2*sol[2*ncli+i]+3*sol[3*ncli+i])
      if(server > nser and freecap[server] >= requests[i]):
         soliter[i] = server
         freecap[server] -= requests[i]
         z += costs[server,i]
         fAssigned = True
         break

      # trying to patch things up
      isFeasible = False
      srvCosts = costs[:,i]
      indc = np.argsort(srvCosts)
      for k in np.random.permutation(len(indc)):
         if (freecap[indc[k]] >= requests[i]):
            soliter[i] = indc[k]
            freecap[indc[k]] -= requests[i]
            z += costs[indc[k], i]
            fAssigned = True
            break
      if not fAssigned:
         print(">>>>>>>>>>>>>>>>>>>>>>>> ASSIGNMENT ERROR <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

   if isFeasible:
      print(f"Feasible solution, cost {z}")
      z = LS.opt10(costs,cap,requests,soliter)
      z = LS.opt11(costs,cap,requests,soliter)
   return isFeasible, soliter, z

def subProblemRelaxCap(requests, costs, cap, b, vlambda):
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

def subgradientRelaxCap(requests, costs, cap, b, alpha=0.1, niter=3, maxuseless=100, minalpha = 0.01):
   zub = np.infty
   tstart = time.time()
   flog = open("log.csv", "w")
   nuseless  = 0  # number of non improving iterations
   alphainit = alpha
   vlambda   = np.zeros(nser)
   iter = 0
   zlb  = 0
   while(iter < niter):
      print(f"SUBGR ===================== iter {iter}")
      (zliter,sol) = subProblemRelaxCap(requests, costs, cap, b, vlambda)
      (isFeas, subgrad, _ , z) = checkFeas(sol,cap,requests,costs)
      isFeasible, soliter, zubiter = computeFOval(sol, costs, requests, cap)
      #fout.write(f"{np.array2string(soliter, max_line_width=10000, separator=',')}\n")
      if isFeasible != isFeas: print(">>>>>>>>>>>>>>>>> FEASIBILITY MISMATCH <<<<<<<<<<<<<<<<<<<")
      nuseless += 1
      if zliter > zlb:   # update lb
         zlb = zliter    # SHOULD SAVE THE BEST LB SOLUTION HERE
         nuseless = 0
      if(isFeas):        # check for optimality
         if zubiter < zub:  # update ub
            zub = zubiter   # SHOULD SAVE THE BEST UB SOLUTION HERE
            nuseless = 0
            tnow = time.time()
            print(f" ---- NEW ZUB: {zub} time {tnow - tstart}")
         isOpt = True
         for i in np.arange(len(subgrad)):
            if(vlambda[i]!=0 and subgrad[i]!=0):
               isOpt = False
         if isOpt:
            print(f"Trovato l'ottimo! zopt = zlb = {zlb}")
            return (zlb,sol)
      sub2 = 0           # not provably optimal
      for i in np.arange(nser): sub2 += subgrad[i]*subgrad[i]
      zz = zlb*2
      step = alpha*(min(zub,zz) - zlb)/sub2
      for i in np.arange(nser):
         vlambda[i] += step * subgrad[i]
         if(vlambda[i]<=0): vlambda[i]=0
      tnow = time.time()
      print(f"subgr, iter {iter} zlb= {zlb} zliter={zliter} zubiter={zubiter} zub={zub} step = {step} time {tnow - tstart}")
      #print(f"Lambda {vlambda}")
      #print(f"Subgr  {subgrad}")
      flog.write(f"{iter},{zlb},{zliter},{zubiter},{zub}\n")
      iter += 1
      if(iter%100 == 0):
         alpha = 0.8*alpha
         if alpha < minalpha:
            alpha = minalpha
      if nuseless > maxuseless:
         nuseless = 0
         alpha = alphainit
         vlambda = 20*np.random.random(nser)

   flog.close()
   return (zlb,sol)

def subProblemRelaxAss(req, costs, cap, b, lmbda):
   ncol = 2*ncli*nser
   nx   = ncli*nser
   categx ='Binary'  # 'Continuous'
   X = pulp.LpVariable.dicts('X%s', (range(nx)),
                        cat=categx,
                        lowBound=0,
                        upBound=1)

   categq ='Integer'  # 'Continuous'
   Q = pulp.LpVariable.dicts('Q%s', (range(nx,ncol)),
                        cat=categq,
                        lowBound=0,
                        upBound=max(req))
   X.update(Q) # append Q to X

   # create the LP object, set up as a MINIMIZATION problem
   probl = pulp.LpProblem('GDO', pulp.LpMinimize)

   # -------------------------------- cost function section
   c = np.zeros(ncol)
   # costi x
   c[0:nx] = [(costs[i,j] - lmbda[i,j]*req[j]) for i in np.arange(nser) for j in np.arange(ncli)]
   # costi q
   for i in np.arange(nser):
      for j in np.arange(ncli):
         c[nx+i*ncli+j] = lmbda[i,j]

   probl += sum(c[i] * X[i] for i in range(ncol))

   # -------------------------------- constraint section
   nrows = 0
   # amount constraint Sum xij leq bj
   for j in np.arange(ncli):
      probl += sum(X[i*ncli+j] for i in np.arange(0,nser)) == b[j], f"b{nrows}"
      nrows += 1

   # client request constraints Sum qij = reqj
   for j in np.arange(ncli):
      probl += sum(X[nx+i*ncli+j] for i in np.arange(0,nser)) == req[j], f"ass{j}"
      nrows += 1

   # capacity constraints
   for i in range(nser):
      probl += sum(X[nx+i*ncli+j] for j in range(0,ncli)) <= cap[i], f"cap{i}"
      nrows += 1

   # save the model in a lp file
   #probl.writeLP("subr2.lp")

   # solve the model
   probl.solve(pulp.PULP_CBC_CMD(msg=0))
   cost = pulp.value(probl.objective)
   print(f"Subpr. status: {pulp.LpStatus[probl.status]} cost {cost}")
   # variable values
   sol = np.zeros(ncol)
   lstsol = []
   for i in np.arange(ncol):
      v = probl.variables()[i]
      if (v.varValue > 0):
         ii = int(v.name[1:])
         sol[ii]=v.varValue
         lstsol.append({'cli': ii%ncli, 'ser': ii//ncli})
         #print(f"{v} = {v.varValue}  i: {ii}  cli {ii%ncli}, ser: {ii//ncli}")
   #print(f"LR sol: {sol}")

   return (cost,sol)

def subgradientRelaxAss(requests, costs, cap, b, alpha=0.1, niter=3, maxuseless=100, minalpha = 0.01):
   zub = np.infty
   tstart = time.time()
   flog = open("log.csv", "w")
   nuseless  = 0  # number of non improving iterations
   alphainit = alpha
   lmbda   = np.zeros(ncli*nser).reshape(nser,ncli)
   iter = 0
   zlb  = 0
   while(iter < niter):
      print(f"SUBGR ===================== iter {iter}")
      zubiter = np.infty
      (zliter,sol) = subProblemRelaxAss(requests, costs, cap, b, lmbda)
      (isFeas, _, subgrad, z) = checkFeas(sol,cap,requests,costs)
      if(isFeas):
         isFeasible, soliter, zubiter = computeFOval(sol, costs, requests, cap)

      # update of lb,ub
      nuseless += 1
      if zliter > zlb:   # update lb
         zlb = zliter    # SHOULD SAVE THE BEST LB SOLUTION HERE
         nuseless = 0
      if(isFeas):        # check for optimality
         if zubiter < zub:  # update ub
            zub = zubiter   # SHOULD SAVE THE BEST UB SOLUTION HERE
            nuseless = 0
            tnow = time.time()
            print(f" ---- NEW ZUB: {zub} time {tnow - tstart}")

         # check for optimality
         isOpt = True
         for i in np.arange(len(subgrad)):
            if(lmbda[i]!=0 and subgrad[i]!=0):
               isOpt = False
         if isOpt:
            print(f"Trovato l'ottimo! zopt = zlb = {zlb}")
            return (zlb,sol)

      # penalty update
      sub2 = 0           # not provably optimal
      for i in range(len(subgrad)): sub2 += subgrad[i]*subgrad[i]
      zz = zlb*1.5 #2
      step = alpha*(min(zub,zz) - zlb)/sub2
      for i in np.arange(nser):
         for j in np.arange(ncli):
            lmbda[i,j] += step * subgrad[i*ncli + j]
            if(lmbda[i,j]<=0): lmbda[i,j]=0
      tnow = time.time()
      print(f"subgr, iter {iter} zlb= {zlb} zliter={zliter} zubiter={zubiter} zub={zub} step = {step} time {tnow - tstart}")
      #print(f"Lambda {lmbda}")
      #print(f"Subgr  {subgrad}")
      flog.write(f"{iter},{zlb},{zliter},{zubiter},{zub}\n")
      iter += 1
      if(iter%100 == 0):
         alpha = 0.8*alpha
         if alpha < minalpha:
            alpha = minalpha
      if nuseless > maxuseless:
         nuseless = 0
         alpha = alphainit
         lmbda = np.zeros(ncli * nser).reshape(nser, ncli)
         for i in range(nser):
            for j in range(ncli):
               r = np.random.random()
               if (r > 0.1): lmbda[i, j] = 1

   flog.close()
   return (zlb,sol)

def readData(name):
   name = f"../generator/{name}"
   with open(f'{name}.json', 'r') as file:
      dct = json.load(file)

   name = dct["name"]
   n = dct["n"]
   m = dct["m"]
   cols = dct["cols"]
   rows = dct["rows"]
   req = dct["req"]
   cap = dct["cap"]
   qcost = dct["qcost"]
   b   = dct["b"]
   df2 = pd.read_csv('../generator/seedMatrix.csv',header=None, skiprows=1)
   df3 = df2.T.loc[rows,cols]
   df3 = df3.reset_index(drop=True)
   df3.columns = range(df3.shape[1])
   return df3.values,qcost,req,cap,b

if __name__ == "__main__":
   global zub
   zub   = np.infty
   niter = 0

   # old code, calls the lagrangian heuristic. To be updated, in case of use
   isOld = False
   if(isOld):
      with open('config.json') as jconf:
         conf = json.load(jconf)
      print(conf)
      niter = conf['niter']
      alpha = conf['alpha']
      minalpha = conf['minalpha']
      maxuseless = conf['maxuseless']
      numdouble = conf["numdouble"]
      fCapAss = conf["fCapAss"]

      dfcosts = pd.read_csv("costs.csv")
      dfreq   = pd.read_csv("requests.csv")

      ncli = dfcosts.shape[1]
      nser = dfcosts.shape[0]
      b = np.ones(ncli)
      # generate numdouble double assignment requests
      for i in range(numdouble):
         while True:
            id = random.randint(0,ncli-1)
            if b[id] == 1: break
         b[id] = 2

      costs = dfcosts.iloc[0:nser, 0:ncli].values
      req   = dfreq.iloc[0,0:ncli].values
      cap   = dfreq.iloc[0:nser,ncli].values

      if(niter>0):
         if(fCapAss==0):
            (zLR,sol) =  subgradientRelaxCap(req,costs,cap,b,
                                    alpha=alpha, niter = niter, maxuseless=maxuseless, minalpha=minalpha)
         else:
            (zLR, sol) = subgradientRelaxAss(req,costs,cap, b,
                                             alpha=alpha, niter=niter, maxuseless=maxuseless, minalpha=minalpha)
         print(f"lagrangian model, cost {zLR}")

   for inst in range(5):
      print(f"-------------------------- solving inst = {inst}")
      name = f"inst_52_4_0_{inst}"
      costs,qcost,req,cap,b = readData(name)
      ncli = len(req)
      nser = len(cap)
      nmult = np.sum(b != 1)
      row_averages = np.mean(costs, axis=1)

      tstart = time.process_time()

      fOptimal = True
      if fOptimal:
         (lbcost,sol) =  makeMIPmodel(costs,qcost,req,cap,b,isInteger=False)
         print(f"LP model, cost {lbcost}")
         tlp = time.process_time()
         print("cplex LP tcpu in seconds:", tlp - tstart)

         (optcost,sol) =  makeMIPmodel(costs,qcost,req,cap,b,isInteger=True)
         print(f"IP model, cost {optcost}")
         tmip = time.process_time()
         print("cplex IP tcpu in seconds:", tmip - tlp)

      fHexaly = True    # use hexaly local solver
      if fHexaly:
         hexlb,hexub = LS.hexalyLocSearch(costs,qcost,req,cap,b,time_limit = int(tmip - tlp + 1))
      thex = time.process_time()
      print("hexaly tcpu in seconds:", thex - tmip)

      tend = time.process_time()
      print("tcpu in seconds:", tend - tstart)
      fout = open("solutions.txt", "a")
      fout.write(f"{name},{ncli},{nser},{nmult},{hexlb},{hexub},{thex-tmip},{lbcost},{tlp-tstart},{optcost},{tmip-tlp}\n")
      fout.close()
   print("fine")