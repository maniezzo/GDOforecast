#include "GDOforenback.h"


// file print a 1D array of ints
void fprintIntArray(ofstream& flog, vector<int> a, int n)
{  int i;
   for (i = 0; i < n; i++)
      flog << a[i] << " ";
   flog << endl;
}

int read_data()
{  int i,j;
   string infile, line;

   infile = "milanoGAP_1.json";

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

   c = vector<vector<int>>(m, vector<int> (n,0));
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         c[i][j] = JSV["cost"][i][j];

   req = vector<vector<int>>(m, vector<int>(n, 0));
   for (i = 0; i < m; i++)
      for (j = 0; j < n; j++)
         req[i][j] = JSV["req"][i][j];

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

   //if (!isForward && j < 0)           // reached complete solutions (no dad)
   //{  if (stack[currNode].z < zub)
   //      z = readSolutionB(flog, currNode, indCost);
   //   goto end;
   //}

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

      if ((stack[currNode].capused[i] + req[i][j]) <= cap[i] &&
         (stack[currNode].z + c[i][j]) < zub)
      {
         node newNode;
         for (int ii = 0; ii < m; ii++) newNode.capused.push_back(stack[currNode].capused[ii]);
         newNode.capused[i] = stack[currNode].capused[i] + req[i][j];
         newNode.dad = currNode;
         newNode.client = j;
         newNode.server = i;
         newNode.z = stack[currNode].z + c[i][j];
         stack.push_back(newNode);
         indLastNode++;
         if (isForward)
         {  insertInOrder(fList[jlev+1], indLastNode);   // inserts the index of the node in the level list
            //checkMatch(flog, iter, jlev + 1, indLastNode, isForward, indCost);       // check for feasible completion
            if(isVerbose)   
               flog << "lev." << jlev+1 << " fromnode " << currNode << " tonode " << indLastNode << " cost " << newNode.z << endl;
         }
         else
         {  insertInOrder(bList[jlev-1], indLastNode);   // inserts the index of the node in the level list
            //checkMatch(flog, iter, jlev - 1, indLastNode, isForward, indCost);       // check for feasible completion
            if(isVerbose)
               flog << "lev." << jlev-1 << " fromnode " << currNode << " tonode " << indLastNode << " cost " << newNode.z << endl;
         }
         numNewNodes++;
      }
      ii++;
   }
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

// finds the next node to expand forward
int findNextNodeF(int jlev, int newNodes, int openNodes)
{  int jmin;

   if (newNodes > 0 || jlev == n-1)    // if tree height can be increased, go on
      jlev++;
   else              // find the highest level with unexpanded nodes
   {  for (jmin = 0; jmin < n; jmin++)
         if (fList[jmin].size() > 0)
            break;
      jlev = jmin;
   }
   return jlev;
}

// one run of forward beam search
int sweepForward(ofstream& flog, int iter, vector<vector<int>> c, int delta, int maxNodes, int openNodes, vector<int> indCost)
{  int j, jlev, k;
   int currNode, newNodes, numExp;
   int nNodes0 = indLastNode;

   jlev = newNodes = currNode = 0;             // got to initialize them 
   while (jlev < n)                               // main construction loop, could be while true
   {
      jlev = findNextNodeF(jlev, newNodes, openNodes);     // backjunmping!
      if (jlev == n || (indLastNode - nNodes0) > (maxNodes / 10))
         break;

      newNodes = 0;                            // new nodes at corrent level
      for (k = 0; k < delta && fList[jlev].size()>0; k++)       // EXPANSION
      {
         currNode = fList[jlev].front();                 // gets the first element
         if (jlev < n && stack[currNode].z < zub)
         {  j = (jlev == n - 1 ? -1 : indCost[jlev + 1]);    // client order by regrets
            numExp = expandNode(flog, iter, c, j, jlev, currNode, indCost, true);
            openNodes += numExp;
            newNodes += numExp;
         }
         else
            numFathomed++;
         if (indLastNode > maxNodes)
         {  cout << "node limit reached" << endl;
            goto end;
         }
         fTree[jlev].push_back(currNode);       // append to list of expanded nodes
         fList[jlev].pop_front();               // remove from list of nodes to expand
         openNodes--;
         if (stack[currNode].z > fTopCost[jlev])
            fTopCost[jlev] = stack[currNode].z; // update max cost of expanded node at the level
         else
            if (isVerbose)
               if (stack[currNode].z < fTopCost[jlev]) cout << "[sweepForward] inner cost insertion" << endl;
      }
      if (isVerbose)
      {  cout << "[sweepForward] iter " << iter << " Level " << jlev << " expanded " << k << " new nodes " << newNodes << " open nodes " << openNodes << " tot nodes " << indLastNode << " top cost " << fTopCost[jlev] << " zub " << zub << endl;
         flog << "[sweepForward] iter " << iter << " Level " << jlev << " expanded " << k << " new " << newNodes << " open " << openNodes << " tot " << indLastNode << " topcost " << fTopCost[jlev] << " fathomed " << numFathomed << endl;
      }
   }
end:
   return openNodes;
}

int goFnB()
{  int i,j;
   int zub0, openNodesF,openNodesB,nNoImproved;
   int z,iter,currNode;

   // -------------------- log file
   ofstream flog;
   flog.open("foreandback.log");
   flog << fixed << setprecision(3);

   // -------------------- initializaations
   for (j = 0; j < n; j++)
   {  fTree.push_back(list<int>());
      bTree.push_back(list<int>());
      fList.push_back(list<int>());
      bList.push_back(list<int>());
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
   indLastNode = 1;
   zub = zub0 = INT_MAX;
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
      indLastNode < maxNodes &&
      iter < maxIter &&
      nNoImproved < 2)
   {
      zub0 = zub;
      nNoImproved++; // iterations without zub improvement
      cout << "Iter " << iter << " Num nodes: " << indLastNode << " open nodes forw." << openNodesF << " open nodes backw." << openNodesB << endl;
      cout << " ---------- FORWARD -------------> " << endl;
      openNodesF = sweepForward(flog, iter, c, delta, maxNodes, openNodesF, indCost);
      cout << " <---------- BACKWARD ------------ " << endl;
      //openNodesB = sweepBackward(flog, iter, c, delta, maxNodes, openNodesB, indCost);
      iter++;
      if (zub < zub0) nNoImproved = 0; // zub improved in current iteration
   }
   if (indLastNode >= maxNodes) cout << "maxNodes exceeded" << endl;
   if (iter >= maxIter) cout << "maxIter exceeded" << endl;
   if (nNoImproved >= 2) cout << "no improvements for 2 iterations, no hope" << endl;

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
   for (i = 0; i < m; i++) capused[i] = 0;

   // controllo assegnamenti
   for (j = 0; j < n; j++)
      if (sol[j] < 0 || sol[j] >= m)
      {  cost = INT_MAX;
         goto lend;
      }
      else
         cost += c[sol[j]][j];

   // controllo capacità
   for (j = 0; j < n; j++)
   {  capused[sol[j]] += req[sol[j]][j];
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
   isVerbose = false;
   maxNodes  = 10000000;
   maxIter   = 1000;
   delta     = 5;    // num offspring
   read_data();
   goFnB();

   cout << "<ENTER> to exit ..." << endl;
}
