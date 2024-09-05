import numpy as np, pandas as pd

# tries each client with each other facility
def opt10(int** c, bool isOriginal)
   int z=0,zorg
   int i, isol, j
   vector<int> capleft(m)

   for(i=0i<mi++) capleft[i] = GAP->cap[i]
   for (j = 0 j < n j++)
      capleft[sol[j]] -= req[sol[j]][j]
      z += c[sol[j]][j]

   zorg=z

l0:
   for (j = 0 j < n j++)
      isol = sol[j]
      for (i = 0 i < m i++)
         if (i == isol): continue
         if (c[i][j] < c[isol][j] && capleft[i] >= req[i][j])
            # remove from isol and assign to i
            sol[j] = i
            capleft[i]    -= req[i][j]
            capleft[isol] += req[isol][j]
            z -= (c[isol][j] - c[i][j])
            if(isOriginal and z<zub):
               GAP->storeBest(sol,z)
               cout << "[1-0 opt] new zub " << zub << endl

            goto l0
   #if(z<zorg)
   #   cout << "2opt improved" << endl
   return z


# scambio assegnamento fra due clienti
def opt11(int** c, bool isOriginal)
   int i,j,j1,j2,temp,cap1,cap2
   int delta, z=0, zcheck, zorg
   vector<int> capleft(m)

   for(i=0i<mi++) capleft[i] = GAP->cap[i]
   for (j = 0 j < n j++)
      capleft[sol[j]] -= req[sol[j]][j]
      z += c[sol[j]][j]

   zcheck = GAP->checkSol(sol)
   zorg = z

l0:
   for(j1=0j1<nj1++)
      for(j2=j1+1j2<nj2++)
         delta = (c[sol[j1]][j1] + c[sol[j2]][j2]) - (c[sol[j1]][j2] + c[sol[j2]][j1])
         if(delta > 0)
            cap1 = capleft[sol[j1]] + req[sol[j1]][j1] - req[sol[j1]][j2]
            cap2 = capleft[sol[j2]] + req[sol[j2]][j2] - req[sol[j2]][j1]
            if(cap1>=0 && cap2 >=0)
               capleft[sol[j1]] += req[sol[j1]][j1] - req[sol[j1]][j2]
               capleft[sol[j2]] += req[sol[j2]][j2] - req[sol[j2]][j1]
               temp    = sol[j1]
               sol[j1] = sol[j2]
               sol[j2] = temp
               z -= delta
               zcheck = GAP->checkSol(sol)
               if(isOriginal)
                  if(abs(z-zcheck) > GAP->EPS)
                     cout << "[1-1] ohi" << endl
                  else if(z<zub)
                     zub = z
                     for (int k = 0 k < n k++) solbest[k] = sol[k]
                     cout << "[1-1 opt] new zub " << zub << endl
               goto l0

   zcheck = 0
   for(j=0j<nj++):
      zcheck += c[sol[j]][j]
   if(abs(zcheck - z) > GAP->EPS):
      cout << "[1.1opt] Ahi ahi" << endl
   zcheck = checkSol(sol)
   #if (z < zorg)
   #   cout << "2opt improved" << endl
   return z
