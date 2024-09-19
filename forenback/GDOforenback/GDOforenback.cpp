#include "GDOforenback.h"

// file print a 1D array of ints
void fprintIntArray(ofstream& flog, vector<int> a, int n)
{  int i;
   for (i = 0; i < n; i++)
      flog << a[i] << " ";
   flog << endl;
}

int read_data(string infile)
{  int i,j;
   string line;

   try
   {
      cout << "Reading " << infile << endl;
      ifstream jData;
      jData.open(infile, std::ifstream::in);
      jData.exceptions(ifstream::eofbit | ifstream::failbit | ifstream::badbit);
      getline(jData, line);
      //cout << "line:" << line << endl;
      jData.close();
   }
   catch (std::exception const& e)
   {  cout << "Error: " << e.what() << endl;
      return -1;
   }

   json::Value JSV = json::Deserialize(line);
   string name = JSV["name"];
   n = JSV["numcli"];
   m = JSV["numserv"];

   for (i = 0; i < JSV["cap"].size(); i++)
      cap.push_back(JSV["cap"][i]);

   for (j = 0; j < n; j++)
      req.push_back(JSV["req"][j]);

   for (j = 0; j < n; j++)
      b.push_back(JSV["split"][j]);

   c = vector<vector<int>>(m, vector<int> (n,0));
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         c[i][j] = JSV["cost"][i][j];

   zub = INT32_MAX;
   sol     = vector<int>(n, -1);
   solbest = vector<int>(n, -1);

   cout << "JSON data read" << endl;
   return 0;
}

// c: costs; j: current client; jlev; current tree level; corrNode: node currently expanded (will be father)
int expandNode(ofstream& flog, int iter, vector<vector<int>> c, int j, int jlev, int currNode, vector<int> indCost, bool isForward)
{  int i, ii, numNewNodes = 0, z;
   vector<int> cost(m), indReq(m);
   auto compCost = [&cost](int a, int b) { return cost[a] < cost[b]; };  // ASC order

   if (isForward && j < 0)            // reached complete solutions (no dad)
   {  if (stack[currNode].z < zub)
         z = readSolutionF(flog, currNode, indCost);
      goto end;
   }

   if (!isForward && j < 0)           // reached complete solutions (no dad)
   {  if (stack[currNode].z < zub)
         z = readSolutionB(flog, currNode, indCost);
      goto end;
   }

   // cost of expansions
   for (i = 0; i < m; i++)
   {  cost[i] = c[i][j];
      indReq[i] = i;
   }
   std::sort(indReq.begin(), indReq.end(), compCost);

   // expansion of a node
   ii = numNewNodes = 0;
   while (ii < m)
   {  i = indReq[ii];

      if ((stack[currNode].capused[i] + req[j]) <= cap[i] &&
         (stack[currNode].z + c[i][j]) < zub)
      {
         node newNode;
         for (int ii = 0; ii < m; ii++) newNode.capused.push_back(stack[currNode].capused[ii]);
         newNode.capused[i] = stack[currNode].capused[i] + req[j];
         newNode.dad = currNode;
         newNode.client = j;
         newNode.server = i;
         newNode.z = stack[currNode].z + c[i][j];
         newNode.expanded = false;
         stack.push_back(newNode);
         numNodes++;
         if (isForward)
         {  insertInOrder(fLstUnexp[jlev+1], numNodes);   // inserts the index of the node in the level list
            checkMatch(flog, iter, jlev + 1, numNodes, isForward, indCost);       // check for feasible completion
            if(isVerbose)   
               flog << "lev." << jlev+1 << " fromnode " << currNode << " tonode " << numNodes << " cost " << newNode.z << endl;
         }
         else
         {  insertInOrder(bLstUnexp[jlev-1], numNodes);   // inserts the index of the node in the level list
            checkMatch(flog, iter, jlev - 1, numNodes, isForward, indCost);       // check for feasible completion
            if(isVerbose)
               flog << "lev." << jlev-1 << " fromnode " << currNode << " tonode " << numNodes << " cost " << newNode.z << endl;
         }
         numNewNodes++;
      }
      ii++;
   }
   stack[currNode].expanded = true;
end:
   return numNewNodes;  // new nodes opened at the level 
}

// reads the solutions from a last node of the forward tree
int readSolutionF(ofstream& flog, int currNode, vector<int> indCost)
{  int res = 0, j, jlev, solNode;
   int z, zsol = 0;

   if (stack[currNode].z < zub)
   {
      z = stack[currNode].z;
      // reconstruction of the solution
      solNode = currNode;
      jlev = n - 1;
      while (stack[solNode].server > -1)
      {
         j = indCost[jlev];
         if (j != stack[solNode].client)
            cout << "ouch!" << endl;
         sol[j] = stack[solNode].server;
         zsol += c[sol[j]][j];   // for checking
         flog << "node " << solNode << " j " << j << " i " << sol[j] << " c " << c[sol[j]][j] << " z " << stack[solNode].z;
         flog << " jlev " << jlev << " zsol " << zsol << endl;
         solNode = stack[solNode].dad;
         jlev--;
      }
      fprintIntArray(flog, sol, n);

      if (stack[currNode].z < zub)
      {
         zub = z;
         cout << "New zub: " << zub << endl;
         flog << "FNew zub: " << zub << endl;
         for (j = 0; j < n; j++)
            solbest[j] = sol[j];
      }
   }

   return res;
}

// reads the solutions from a last node of the backward tree
int readSolutionB(ofstream& flog, int currNode, vector<int> indCost)
{  int res = 0, j, jlev, solNode;

   if (stack[currNode].z < zub)
   {  zub = stack[currNode].z;
      // reconstruction of the solution
      solNode = currNode;
      jlev = 0;
      while (jlev < n)
      {  j = indCost[jlev];
         sol[j] = solbest[j] = stack[solNode].server;
         //cout << "node " << solNode << " j " << j << " i " << sol[j] << " c " << GAP->c[sol[j]][j] << " z " << stack[solNode].z << endl;
         solNode = stack[solNode].dad;
         jlev++;
      }
      cout << "New zub: " << zub << endl;
      flog << "BNew zub: " << zub << " "; fprintIntArray(flog, solbest, n);
   }

   return res;
}

// reads a solution as a mix of forw and a backw partials, jLevF last level of forward tree
int readSolutionFB(ofstream& flog, int jLevF, int fNode, int bNode, vector<int> indCost)
{  int res = 0, z = 0, j, jlev, solNode;
   int zsol;

   zsol = stack[fNode].z + stack[bNode].z;

   // reconstruction of the forward part of the solution
   solNode = fNode;
   jlev = jLevF;
   while (jlev > -1 && stack[solNode].server >= 0)
   {  j = indCost[jlev];
      sol[j] = stack[solNode].server;
      z += c[sol[j]][j];
      if(isVerbose) cout << "node " << solNode << " lev " << jlev << " j " << j << " i " << sol[j] << " c " << c[sol[j]][j] << " z " << stack[solNode].z << " ztot " << z << endl;
      solNode = stack[solNode].dad;
      jlev--;
   }

   // reconstruction of the backward part of the solution
   solNode = bNode;
   jlev = jLevF + 1;
   while (jlev < n && stack[solNode].server >= 0)
   {  j = indCost[jlev];
      sol[j] = stack[solNode].server;
      z += c[sol[j]][j];
      if(isVerbose) cout << "node " << solNode << " lev " << jlev << " j " << j << " i " << sol[j] << " c " << c[sol[j]][j] << " z " << stack[solNode].z << " ztot " << z << endl;
      solNode = stack[solNode].dad;
      jlev++;
   }

   flog << "zsol " << zsol << " "; fprintIntArray(flog, sol, n);

   if (z < zub)
   {  zub = z;
      cout << "New zub: " << zub << endl;
      for (j = 0; j < n; j++) solbest[j] = sol[j];
   }

   return z;
}

// insert in list (for- or back-ward). level: level where to insert; elem: element to insert (key: node cost)
int insertInOrder(list<int>& lst, int elem)
{  int res = 0;
   list<int>::iterator it;

   if (lst.size() == 0)
      lst.push_back(elem);
   else
   {  it = lst.begin();
      while (it != lst.end() && stack[*it].z < stack[elem].z)
         ++it;
      lst.insert(it, elem);
   }
   return res;
}

// finds the next node to expand forward (first level with unexpanded nodes)
int findNextNodeF(int jlev, int newNodes, int openNodes)
{  int jmin;

   if (newNodes > 0 || jlev == n-1)    // if tree height can be increased, go on
      jlev++;
   else              // find the highest level with unexpanded nodes
   {  for (jmin = 0; jmin < n; jmin++)
         if (fLstUnexp[jmin].size() > 0)
            break;
      jlev = jmin;
   }
   return jlev;
}

// finds the next node to expand backward (first level with unexpanded nodes)
int findNextNodeB(int jlev, int newNodes, int openNodes)
{  int jmin;

   if (newNodes > 0 || jlev == 0)    // if there were expansions, go on
      jlev--;
   else              // find the lowest level with unexpanded nodes
   {  for (jmin = n - 1; jmin > 0; jmin--)
         if (bLstUnexp[jmin].size() > 0)
            break;
      if (jlev < n - 1)
         jlev = -1;
      else
         jlev = jmin;
   }
   return jlev;
}

// one run of forward beam search
int sweepForward(ofstream& flog, int iter, vector<vector<int>> c, int delta, int maxNodes, int openNodes, vector<int> indCost)
{  int j, jlev, k;
   int currNode, newNodes, numExp;
   int nNodes0 = numNodes;

   jlev = newNodes = currNode = 0;             // got to initialize them 
   while (jlev < n)                            // main construction loop, could be while true
   {
      jlev = findNextNodeF(jlev, newNodes, openNodes);     // backjunmping!
      if (jlev == n || (numNodes - nNodes0) > (maxNodes / 10))
         break;

      newNodes = 0;                            // new nodes at corrent level
      for (k = 0; k < delta && fLstUnexp[jlev].size()>0; k++) // -------------------------- EXPANSION
      {
         currNode = fLstUnexp[jlev].front();                  // gets the first element of unexpanded list
         if (jlev < n && stack[currNode].z < zub)
         {  j = (jlev == n - 1 ? -1 : indCost[jlev + 1]);     // client order by regrets
            numExp = expandNode(flog, iter, c, j, jlev, currNode, indCost, true);
            openNodes += numExp;
            newNodes  += numExp;
         }
         else
            numFathomed++;
         if (numNodes > maxNodes)
         {  cout << "node limit reached" << endl;
            goto end;
         }
         fTree[jlev].push_back(currNode);       // append to list of expanded nodes
         fLstUnexp[jlev].pop_front();               // remove from list of nodes to expand
         openNodes--;
         if (stack[currNode].z > fTopCost[jlev])
            fTopCost[jlev] = stack[currNode].z; // update max cost of expanded node at the level
         else
            if (isVerbose)
               if (stack[currNode].z < fTopCost[jlev]) cout << "[sweepForward] inner cost insertion" << endl;
      }
      if (isVerbose)
      {  cout << "[sweepForward] iter " << iter << " Level " << jlev << " expanded " << k << " new nodes " << newNodes << " open nodes " << openNodes << " tot nodes " << numNodes << " top cost " << fTopCost[jlev] << " zub " << zub << endl;
         flog << "[sweepForward] iter " << iter << " Level " << jlev << " expanded " << k << " new " << newNodes << " open " << openNodes << " tot " << numNodes << " topcost " << fTopCost[jlev] << " fathomed " << numFathomed << endl;
      }
   }
end:
   return openNodes;
}

// one run of backward beam search
int sweepBackward(ofstream& flog, int iter, vector<vector<int>> c, int delta, int maxNodes, int openNodes, vector<int> indCost)
{  int j, jlev, k;
   int currNode, newNodes = 0, numExp;

   jlev = n - 1;
   while (jlev >= 0)                                // main construction loop
   {  jlev = findNextNodeB(jlev, newNodes, openNodes); // backjunmping!
      if (jlev < 0)
      {  break;
         flog << "[sweepBackward] Level " << jlev << " expanded " << k << " new nodes " << newNodes << " open nodes " << openNodes << " tot nodes " << numNodes << " top cost " << bTopCost[jlev] << endl;
      }

      newNodes = 0;
      for (k = 0; k < delta && bLstUnexp[jlev].size()>0; k++)
      {  currNode = bLstUnexp[jlev].front();               // gets the first element
         if (jlev >= 0 && stack[currNode].z < zub)
         {  j = (jlev == 0 ? -1 : indCost[jlev - 1]);    // client order by regrets
            numExp = expandNode(flog, iter, c, j, jlev, currNode, indCost, false);
            openNodes += numExp;
            newNodes += numExp;
         }
         else
            numFathomed++;
         if (numNodes > maxNodes) goto end;   // node limit reached
         bTree[jlev].push_back(currNode);       // append to list of expanded nodes
         bLstUnexp[jlev].pop_front();               // remove from list of nodes to expand
         openNodes--;
         if (stack[currNode].z > bTopCost[jlev])
            bTopCost[jlev] = stack[currNode].z; // update max cost of expanded node at the level
         else
            if (isVerbose)
               if (stack[currNode].z < bTopCost[jlev]) cout << "[sweepBackward] inner cost insertion" << endl;
      }
      if (isVerbose)
      {  cout << "[sweepBackward] iter " << iter << " Level " << jlev << " expanded " << k << " new nodes " << newNodes << " open nodes " << openNodes << " tot nodes " << numNodes << " top cost " << bTopCost[jlev] << endl;
         flog << "[sweepBackward] iter " << iter << " Level " << jlev << " expanded " << k << " new " << newNodes << " open " << openNodes << " tot " << numNodes << " topcost " << bTopCost[jlev] << " numfathomed " << numFathomed << endl;
      }
   }
end:
   return openNodes;
}

// check for matching partial solutions, jlev level of indLastNode
int checkMatch(ofstream& flog, int iter, int jlev, int indLastNode, bool isForward, vector<int> indCost)
{  int i, z = -1, res = 0;
   list<int>* lstCompletions;
   std::list<int>::const_iterator iterator;

   if (jlev == n - 1 || jlev == 0) goto end;

   if (isForward)
      lstCompletions = &bTree[jlev + 1];
   else
      lstCompletions = &fTree[jlev - 1];

   // Iterate and print values of the completion list
   for (iterator = (*lstCompletions).begin(); iterator != (*lstCompletions).end(); ++iterator)
   {  if (stack[*iterator].server < 0)
         continue;
      for (i = 0; i < m; i++)
         if (stack[indLastNode].capused[i] + stack[*iterator].capused[i] >cap[i])
            goto next;
      z = stack[indLastNode].z + stack[*iterator].z;
      if (isVerbose)
      {  cout << "MATCHING FEASIBLE! cost " << z << endl;
         flog << (isForward ? "forw" : "backw") << " iter " << iter << " level " << jlev << " node_compl " << *iterator;
         flog << " cost " << z << " zub " << zub << " ";
      }

      //if(z < zub)
      {  int jLevF = (isForward ? jlev : jlev - 1);
         int fNode = (isForward ? indLastNode : *iterator);
         int bNode = (isForward ? *iterator : indLastNode);

         z = readSolutionFB(flog, jLevF, fNode, bNode, indCost); // with fprintarray
         if (isVerbose)
         {  cout << "f node " << indLastNode << " z_f " << stack[indLastNode].z <<
               " b node " << *iterator << " z_b " << stack[*iterator].z << " zub " << zub << endl;
            flog << "f_node " << indLastNode << " z_f " << stack[indLastNode].z <<
               " b_node " << *iterator << " z_b " << stack[*iterator].z << endl;
         }
      }
   next:;
   }

end:
   return z;
}

// computes the lower bound at the iteration
int computeLB()
{  int i,j,k,z,zb,zlbiter=0;
   list<int>::iterator it;

   // forward pass
   for (j=0; j<n-1; j++)
   {  // at each level, sum of the cost of the most expensive expanded and least expensive unexpanded backward
      z = 0;
      i = fTree[j].back();            // the last element of the list, the most expensive one
      if(stack[i].expanded && stack[i].z > z)
         z = stack[i].z;

      if(bLstUnexp[j+1].size() > 0)
      {  zb = INT_MAX;
         i = bLstUnexp[j+1].front();    // the first element of unexpanded list
         if (!stack[i].expanded && stack[i].z < zb)
            zb = stack[i].z;
         if(zb<INT_MAX) z += zb; // lower bound forward at this level
      }

      if(z>zlbiter) zlbiter = z;
   }

   // backwardpass
   for (j=n-1; j>0; j--)
   {  // at each level, sum of the cost of the most expensive expanded and least expensive unexpanded forward
      z = 0;
      i = bTree[j].back();            // the last element of the list, the most expensive one
      if (stack[i].expanded && stack[i].z > z)
         z = stack[i].z;

      if (fLstUnexp[j-1].size() > 0)
      {  zb = INT_MAX;
         i = fLstUnexp[j-1].front();    // the first element of unexpanded list
         if (!stack[i].expanded && stack[i].z < zb)
            zb = stack[i].z;
         if (zb < INT_MAX) z += zb; // lower bound forward at this level
      }

      if (z > zlbiter) zlbiter = z;
   }

   return zlbiter;
}

int goFnB()
{  int i,j;
   int zub0, openNodesF,openNodesB,nNoImproved;
   int z,iter,zlbIter,currNode;

   // -------------------- log file
   ofstream flog;
   flog.open("foreandback.log");
   flog << fixed << setprecision(3);

   // -------------------- initializaations
   for (j = 0; j < n; j++)
   {  fTree.push_back(list<int>());
      bTree.push_back(list<int>());
      fLstUnexp.push_back(list<int>());
      bLstUnexp.push_back(list<int>());
      fTopCost.push_back(-1);
      bTopCost.push_back(-1);
   }
   capleft = vector<int>(m);
   for (i=0; i<m; i++) capleft[i] = cap[i];

   // ------------------------------------- initialize trees
   node rootF, rootB;
   rootF.z = rootB.z = 0;
   rootF.dad = rootB.dad = 0;
   rootF.server = rootB.server = -1;
   rootF.client = rootB.client = -1;
   rootF.capused.resize(m);
   rootB.capused.resize(m);
   for (i = 0; i < m; i++)
   {  rootF.capused[i] = 0;
      rootB.capused[i] = 0;
   }
   stack.push_back(rootF);
   stack.push_back(rootB);
   numNodes = 1;
   zub = zub0 = INT_MAX;
   zlb = 0;
   openNodesF = openNodesB = nNoImproved = 0;

   vector<int> indCost(n);                   // index of order of expansion of the clients, in case I wonted it
   for (j = 0; j < n; j++) indCost[j] = j;   // support for ordering

   // initialize forward and backward trees
   z = 0;
   iter = 0;
   currNode = 0;
   numFathomed = 0;
   openNodesF += expandNode(flog, iter, c, indCost[0], -1, currNode, indCost, true);   // stack initialization, forward
   fTree[0].push_back(currNode);                // append to forward list of expanded nodes

   currNode = 1;
   openNodesB += expandNode(flog, iter, c, indCost[n-1], n, currNode, indCost, false); // stack initialization, backward
   bTree[n - 1].push_back(currNode);            // append to backward list of expanded nodes

   // ---------------------------------------------- tree expansions, search
   while ((openNodesF + openNodesB) > 0 &&
      numNodes < maxNodes
      && iter < maxIter 
//      && nNoImproved < 2
      )
   {
      zub0 = zub;
      nNoImproved++; // iterations without zub improvement
      cout << " ---------- FORWARD -------------> " << endl;
      openNodesF = sweepForward(flog, iter, c, delta, maxNodes, openNodesF, indCost);
      cout << " <---------- BACKWARD ------------ " << endl;
      openNodesB = sweepBackward(flog, iter, c, delta, maxNodes, openNodesB, indCost);

      zlbIter = computeLB();
      if (zub < zub0) 
      {  nNoImproved = 0; // zub improved in current iteration
         zub = zub0;
      }
      if (zlbIter > zlb) 
      {  nNoImproved = 0; // zlb improved in current iteration
         zlb = zlbIter;
      }
      cout << "Iter " << iter << " Num nodes: " << numNodes << " open forw." << openNodesF << " open backw." << openNodesB << " zlb " << zlb << " zub " << zub << endl;
      iter++;
   }
   if (numNodes >= maxNodes) cout << "maxNodes exceeded" << endl;
   if (iter >= maxIter) cout << "maxIter exceeded" << endl;
//   if (nNoImproved >= 2) cout << "no improvements for 2 iterations, no hope" << endl;

   if (abs(checkSol(solbest) - zub) > 0)
   {  cout << "[forwardBackward]: Error, solution cost mismatch" << endl;
      z = INT_MAX;
   }
   else
      cout << "Construction terminated. zub = " << zub << endl;


   if (flog.is_open()) flog.close();
   return 0;
}

// controllo ammissibilità soluzione
int checkSol(vector<int> sol)
{  int cost = 0;
   int i, j;
   vector<int> capused(m);
   vector<int> bsol(n);
   for (i=0; i<m; i++) capused[i] = 0;
   for (j=0; j<n; j++) bsol[j] = 0;

   // controllo assegnamenti
   for (j = 0; j < n; j++)
      for(i=0;i<m;i++)
         if (sol[j] == i)
         {  bsol[i]++;
            cost += c[sol[j]][j];
         }

   for (j = 0; j < n; j++)
      if(bsol[j]!=b[j])
      {  cost = INT_MAX;
         goto lend;
      }

   // controllo capacità
   for (j = 0; j < n; j++)
   {  capused[sol[j]] += req[j];
      if (capused[sol[j]] > cap[sol[j]])
      {  cost = INT_MAX;
         goto lend;
      }
   }
lend:
   return cost;
}

int main()
{
   string inFile = "GDO_52_0.json";
   isVerbose = false;
   maxNodes  = 10000000;
   maxIter   = 100; 
   delta     = 5;    // num offspring
   read_data(inFile);
   goFnB();

   cout << "<ENTER> to exit ..." << endl;  cin.get();
}
