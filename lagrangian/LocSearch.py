import numpy as np, pandas as pd
import copy

import hexaly.optimizer
import sys

def hexalyLocSearch(cost, qcost, req, cap, b, time_limit = 60):
   n = len(req) # clients
   m = len(cap) # servers
   numVar = n*m
   with hexaly.optimizer.HexalyOptimizer() as optimizer:
      # Declare the optimization model
      lsModel = optimizer.model

      # decision: assignments: client j assigned to server i
      x = [[lsModel.bool() for j in range(n)] for i in range(m)] # assignment client / server
      q = [[lsModel.int(0,int(req[j]))  for j in range(n)] for i in range(m)] # quantity shipped

      # client requests satisfied
      quant = [None] * n # delivered quantities
      for j in range(n):
         quant[j] = lsModel.sum(q[i][j] for i in range(m))
         lsModel.constraint(quant[j] == req[j])

      # assignment
      nass = [None] * n
      for j in range(n):
         nass[j] = lsModel.sum(x[i][j] for i in range(m))
         lsModel.constraint(nass[j] <= b[j])

      # server capacities
      usedcap = [None] * m
      for i in range(m):
         usedcap[i] = lsModel.sum([q[i][j] for j in range(n)])
         lsModel.constraint(usedcap[i] <= cap[i])

      # objective function
      total_cost = lsModel.sum((cost[i][j]*x[i][j]+qcost[i]*q[i][j]) for i in range(m) for j in range(n))
      lsModel.minimize(total_cost)

      #link x - q
      for i in range(m):
         for j in range(n):
            lsModel.constraint(q[i][j] - req[j]*x[i][j] <= 0)

      lsModel.close()
      fWriteModel = False
      if(fWriteModel):
         optimizer.save_environment("lmModel.hxm")

      # ----------------------------------------- go solve!
      optimizer.param.set_time_limit(time_limit)
      optimizer.param.set_verbosity(0)
      optimizer.solve()
      status = optimizer.solution.status.name
      print(f"Solution status: {status}")
      if(status!="OPTIMAL" and status!="FEASIBLE"):
         return -1,-1
      sol = optimizer.get_solution()
      ub  = total_cost.value
      lb  = sol.get_objective_bound(0)
      gap = sol.get_objective_gap(0)
      print(f"Bound {lb} gap {gap}")

      # Write the solution
      print(f"Hexaly: cost {ub}\nSolution:")
   '''      
   for i in range(m):
      for j in range(n):
         if x[i][j].value == 1:
            print(f"i:{i} j:{j} x {x[i][j].value} q {q[i][j].value} cost: {cost[i][j]} req: {req[j]}")
   '''
   return lb,ub

def checkSol(sol,cap,req,costs):
   isFeas = True
   nser = len(cap)
   ncli = len(req)
   subgrad = np.zeros(nser)
   # capacity constraints Sum qij <= capi
   for i in np.arange(nser):
      sum = 0
      for j in np.arange(ncli):
         if(sol[j]==i):
            sum += req[j]
      subgrad[i] = sum - cap[i]
      if sum > cap[i]:
         isFeas = False
   # assignemtn constraints
   for j in np.arange(ncli):
      if(not(sol[j]>=0 and sol[j]<nser)):
         isFeas = False
   # check cost
   z = 0
   for j in np.arange(ncli):
      z += costs[sol[j],j]
   #print(f"Checked cost: {z}")

   return isFeas, z

# tries each allocation (possibly partial) with each other facility
def opt10(c,cap,req,x):
   z=0
   zorg=0
   n = len(req)
   m = len(cap)
   capleft = np.zeros(m)

   for i in range(m):
      capleft[i] = cap[i]
   for j in range(n):
      capleft[x[j]] -= req[j]
      z += c[x[j],j]

   zcurr = z # cost of solution currently in x
   zorg  = z # cost of seed solution

   fRepeat = True
   while fRepeat:
      fRepeat = False # l0
      for j in range(n):
         isol = x[j]
         for i in range(m):
            if (i == isol): continue
            if (c[i][j] < c[isol][j] and capleft[i] >= req[j]):
               # remove from isol and assign to i
               capleft[x[j]] += req[isol]
               capleft[i]    -= req[isol]
               z -= (c[isol][j] - c[i][j])
               if(z<zcurr):
                  print(f"[1-0 opt] new z {z}")
                  zcurr = z

               x[j] = i
               isol = i
               fRepeat=True #goto .l0 # Jumps back to the l0 label
   if(z<zorg):
      print(f"-- opt10 improved {zorg} -> {z} --")
   return z


# scambio assegnamento fra due clienti
def opt11(c,cap,req,x):
   EPS = 0.001
   z=0
   zorg=0
   n = len(req)
   m = len(cap)
   capleft = np.zeros(m)

   for i in range(m):
      capleft[i] = cap[i]
   for j in range(n):
      capleft[x[j]] -= req[j]
      z += c[x[j],j]

   zcurr = z # cost of solution currently in x
   zorg  = z # cost of seed solution

   isFeas,zcheck = checkSol(x,cap,req,c)

   fRepeat = True
   while fRepeat:
      fRepeat = False  # l0
      for j1 in range(n):
         for j2 in range(j1+1,n):
            delta = (c[x[j1],j1] + c[x[j2],j2]) - (c[x[j1],j2] + c[x[j2],j1])
            if(delta > 0):
               cap1 = capleft[x[j1]] + req[j1] - req[j2]
               cap2 = capleft[x[j2]] + req[j2] - req[j1]
               if(cap1>=0 and cap2 >=0):
                  capleft[x[j1]] += req[j1] - req[j2]
                  capleft[x[j2]] += req[j2] - req[j1]
                  temp  = x[j1]
                  x[j1] = x[j2]
                  x[j2] = temp
                  z -= delta
                  isFeas,zcheck = checkSol(x,cap,req,c)
                  if(abs(z-zcheck) > EPS):
                     print("[1-1] ohi")
                  fRepeat=True #goto .l0 # Jumps back to the l0 label

   isFeas,zcheck = checkSol(x,cap,req,c)
   if(abs(zcheck - z) > EPS):
      print("[1.1opt] Ahi ahi")
   if(z<zorg):
      print(f"-- opt11 improved {zorg} -> {z} --")
   return z

# recovers feasibility in case of partial or overassigned solution
def fixSol(infeasSol, zsol, c, cap, req, zub, solbest):
   imin=-1
   n = len(req)
   m = len(cap)

   capres = copy.deepcopy(cap)
   sol    = copy.deepcopy(infeasSol)
   zsol = 0

   while(zsol <= 0):  # to mimik goto end
      # ricalcolo capacità residue. Se sovrassegnato, metto a sol a -1
      for j in range(n):
         if(sol[j]>=0 and (capres[sol[j]] >= req[sol[j]][j])):
            capres[sol[j]] -= req[sol[j]][j]
         else:
            sol[j] = -1

      for j in range(n):
         if(sol[j]>=0):              # correct, do nothing
            zsol += c[sol[j]][j]
            continue

         # reassign i -1
         minreq = np.infty
         imin = -1
         for i in range(m):
            if(capres[i]>=req[i][j] and req[i][j] < minreq):
               minreq = req[i][j]
               imin    = i

         if(imin<0):
            zsol = np.infty
            break           # could not recover feasibility

         sol[j]=imin
         capres[imin] -= req[imin][j]
         zsol += c[imin][j]

      if(zsol<zub):
         for i in range(n): solbest[i]=sol[i]
         zub = zub = zsol
         print(f"[fixSol] -------- zub improved! {zub}")

      for i in range(n): infeasSol[i]=sol[i]

   return zsol


