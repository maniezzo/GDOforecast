import numpy as np, pandas as pd
from goto import with_goto
@with_goto  # You must use this decorator

# tries each allocation (possibly partial) with each other facility
def opt10(c,cap,req,x,q):
   z=0
   zorg=0
   n = len(req)
   m = len(cap)
   capleft = np.zeros(m)

   for i in range(m):
      capleft[i] = cap[i]
   for j in range(n):
      capleft[x[j]] -= q[x[j]][j]
      z += c[x[j],j]

   zcurr=z

   label .l0
   for j in range(n):
      isol = x[j]
      for i in range(m):
         if (i == isol): continue
         if (c[i][j] < c[isol][j] and capleft[i] >= req[i][j]):
            # remove from isol and assign to i
            x[j] = i
            capleft[i]    -= req[i][j]
            capleft[isol] += req[isol][j]
            z -= (c[isol][j] - c[i][j])
            if(z<zcurr):
               print(f"[1-0 opt] new z {z}")
               zcurr = z

            goto .l0 # Jumps back to the l0 label
   if(z<zorg):
      print("-- 2opt improved --")
   return z

'''
# scambio assegnamento fra due clienti
def opt11(int** c, bool isOriginal)
   int i,j,j1,j2,temp,cap1,cap2
   int delta, z=0, zcheck, zorg
   vector<int> capleft(m)

   for(i=0i<mi++) capleft[i] = GAP->cap[i]
   for (j = 0 j < n j++)
      capleft[x[j]] -= req[x[j]][j]
      z += c[x[j]][j]

   zcheck = GAP->checkSol(sol)
   zorg = z

l0:
   for(j1=0j1<nj1++)
      for(j2=j1+1j2<nj2++)
         delta = (c[x[j1]][j1] + c[x[j2]][j2]) - (c[x[j1]][j2] + c[x[j2]][j1])
         if(delta > 0)
            cap1 = capleft[x[j1]] + req[x[j1]][j1] - req[x[j1]][j2]
            cap2 = capleft[x[j2]] + req[x[j2]][j2] - req[x[j2]][j1]
            if(cap1>=0 && cap2 >=0)
               capleft[x[j1]] += req[x[j1]][j1] - req[x[j1]][j2]
               capleft[x[j2]] += req[x[j2]][j2] - req[x[j2]][j1]
               temp    = x[j1]
               x[j1] = x[j2]
               x[j2] = temp
               z -= delta
               zcheck = GAP->checkSol(sol)
               if(isOriginal)
                  if(abs(z-zcheck) > GAP->EPS)
                     cout << "[1-1] ohi" << endl
                  else if(z<zub)
                     zub = z
                     for (int k = 0 k < n k++) solbest[k] = x[k]
                     cout << "[1-1 opt] new zub " << zub << endl
               goto l0

   zcheck = 0
   for(j=0j<nj++):
      zcheck += c[x[j]][j]
   if(abs(zcheck - z) > GAP->EPS):
      cout << "[1.1opt] Ahi ahi" << endl
   zcheck = checkSol(sol)
   #if (z < zorg)
   #   cout << "2opt improved" << endl
   return z
'''