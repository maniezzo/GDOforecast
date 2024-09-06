import numpy as np, pandas as pd

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

   zcheck = GAP->checkSol(sol)

   fRepeat = True
   while fRepeat:
      fRepeat = False  # l0
         for j1 in range(n):
            for j2 in range(j1+1,n):
               delta = (c[x[j1]][j1] + c[x[j2]][j2]) - (c[x[j1]][j2] + c[x[j2]][j1])
               if(delta > 0):
                  cap1 = capleft[x[j1]] + req[x[j1]][j1] - req[x[j1]][j2]
                  cap2 = capleft[x[j2]] + req[x[j2]][j2] - req[x[j2]][j1]
                  if(cap1>=0 and cap2 >=0):
                     capleft[x[j1]] += req[x[j1]][j1] - req[x[j1]][j2]
                     capleft[x[j2]] += req[x[j2]][j2] - req[x[j2]][j1]
                     temp    = x[j1]
                     x[j1] = x[j2]
                     x[j2] = temp
                     z -= delta
                     zcheck = GAP->checkSol(sol)
                     if(isOriginal):
                        if(abs(z-zcheck) > GAP->EPS):
                           print("[1-1] ohi")
                        elif(z<zub):
                           zub = z
                           for k in range(n): solbest[k] = x[k]
                           print(f"[1-1 opt] new zub {zub}")
                     fRepeat=True #goto .l0 # Jumps back to the l0 label

   zcheck = 0
   for j in range(n):
      zcheck += c[x[j]][j]
   if(abs(zcheck - z) > GAP->EPS):
      print("[1.1opt] Ahi ahi")
   zcheck = checkSol(sol)
   if(z<zorg):
      print(f"-- opt11 improved {zorg} -> {z} --")
   return z
