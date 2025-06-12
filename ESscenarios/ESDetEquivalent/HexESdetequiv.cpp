#include "HexESdetequiv.h"

// deterministic equivalent, Exponential Smoothing (Holt Winters) case, Hexaly solver

// Read instance data, generates also the p and the req
void HexStochMIP::readInstance(string& fileName, int numScen, int nboost, int nmult) 
{  string line;
   int i,j,s;

   ifstream infile;
   cout << "Opening " << fileName << endl;
   infile.exceptions(ifstream::failbit | ifstream::badbit);
   infile.open(fileName.c_str());

   std::stringstream buffer;
   buffer << infile.rdbuf();
   line = buffer.str();
   infile.close();

   json::Value JSV = json::Deserialize(line);
   name = JSV["name"];
   n = (int)JSV["n"];
   m = (int)JSV["m"];

   cap.resize(m,0);
   for (i = 0; i < JSV["cap"].size(); i++)
      cap[i] = JSV["cap"][i];

   cols.resize(n);
   for (i = 0; i < JSV["cols"].size(); i++)
      cols[i] = JSV["cols"][i];

   rows.resize(m);
   for (i = 0; i < JSV["rows"].size(); i++)
      rows[i] = JSV["rows"][i];

   b.resize(n);
   vector<int> ind(b.size());   // Vector of indices
   vector<int> rqst(b.size());  // default richieste, per ordinamento solo
   for (i = 0; i < JSV["b"].size(); i++) 
   {  b[i]    = JSV["b"][i];
      rqst[i] = JSV["req"][i];
      ind[i] = i;
   }
   // Sort indices based on comparing elements in rqst in descending order
   sort(ind.begin(), ind.end(), [&rqst](int i1, int i2) { return rqst[i1] > rqst[i2]; });  // Descending order
   // make splittable the nmult clients with higher requests
   for(i=0;i<nmult;i++)
      b[ind[i]] = 2;

   qcost.resize(m);
   for (i = 0; i < JSV["qcost"].size(); i++)
      qcost[i] = JSV["qcost"][i];

   string matrixFile = "../costMatrix.csv";
   cout << "Opening matrix file " << matrixFile << endl;
   infile.exceptions(ifstream::failbit | ifstream::badbit);
   infile.open(matrixFile);
   // Read the file line by line
   getline(infile, line);   // headers
   xAssCost.resize(m);
   for(i=0;i<m;i++)
      xAssCost[i].resize(n);
   j=0;
   while (j<n && getline(infile, line))
   {  stringstream ss(line);
      string value;

      // Split the line by commas
      i=0;
      while (getline(ss, value, ',') && i<m) 
      {  xAssCost[i][j] = stoi(value); // Convert to integer and add to row
         i++;
      }
      j++;
   }      
   infile.close();

   // ----------- probabilities of the scenarios. So far, equiprobable
   p.resize(numScen);
   for(i=0;i<numScen;i++)
      p[i]=1/numScen;      
}

// reads the numserver forecasts for each of the boosted series
int HexStochMIP::readBoostForecasts(string filePath,int nboost,int numScen)
{  string line;
   int i,j,s;
   ifstream infile;

   cout << "Reading " << filePath << endl;
   infile.exceptions(ifstream::failbit | ifstream::badbit);
   infile.open(filePath);
   // Read the file line by line
   boostFcasts.resize(n);
   for(i=0;i<n;i++)
      boostFcasts[i].resize(nboost);

   i=0;
   while (i<n && getline(infile, line))
   {  stringstream ss(line);
      string value;

      // Split the line by commas
      j=0;
      while (getline(ss, value, ',') && j<nboost) 
      {  boostFcasts[i][j] = round(stof(value)); // Convert to integer and add to row
         j++;
      }
      i++;
   }      
   infile.close();

   // ------------ client requests per scenario
   maxReq = 0;
   req.resize(numScen);
   for(s=0;s<numScen;s++)
   {  req[s].resize(n);
      for(j=0;j<n;j++)
      {  req[s][j] = boostFcasts[j][s]; //generateReq(j,nboost);
         if(req[s][j]>maxReq) maxReq = req[s][j];
      }
   }

   return 0;
}

// The tableu for the bscenario case.
int HexStochMIP::populateTableau(CPXENVptr env, CPXLPptr lp, int numScen, double epsCost)
{  int status,numrows,numcols,numnz;
   int i,j,s,currMatBeg,index,qbeg,epsbeg;
   vector<double> obj;
   vector<double> lb;
   vector<double> ub;
   vector<string> colname;
   vector<int>    rmatbeg;
   vector<int>    rmatind;
   vector<double> rmatval;
   vector<double> rhs;
   vector<char>   sense;
   vector<string> rowname;

   status = CPXchgobjsen(env, lp, CPX_MIN);  // Problem is minimization

   // ------------------------------------------------------ variables section

   // Create the columns for x variables
   numcols = 0;
   for(i=0;i<m;i++)
      for(j=0;j<n;j++)
      {  obj.push_back(xAssCost[i][j]); numcols++;  
         lb.push_back(0.0);  
         ub.push_back(1.0); 
         colname.push_back("x"+to_string(i)+"_"+to_string(j));
      }

   epsbeg = numcols; // index of first eps variable

   // Create columns for eps variables
   for(s=0;s<numScen;s++)
      for(j=0;j<n;j++)
      {  obj.push_back(epsCost); numcols++;  
         lb.push_back(0.0);  
         ub.push_back(maxReq); 
         colname.push_back("eps"+to_string(s)+"_"+to_string(j));
      }

   qbeg = numcols; // index of first q variabla

   // Create the columns for q variables
   for(s=0;s<numScen;s++)
      for(i=0;i<m;i++)
         for(j=0;j<n;j++)
         {  obj.push_back(qcost[i]); numcols++;  
            lb.push_back(0.0);  
            ub.push_back(maxReq); 
            colname.push_back("q"+to_string(s)+"_"+to_string(i)+"_"+to_string(j));
         }

   char** cname = new char* [colname.size()];
   for (int index = 0; index < colname.size(); index++)
      cname[index] = const_cast<char*>(colname[index].c_str());
   status = CPXnewcols(env, lp, numcols, &obj[0], &lb[0], &ub[0], NULL, cname);
   delete[] cname;

   if (status)  cout << "ERROR" << endl;

   // ------------------------------------------------------ constraints section

   // quantity q constraints.
   {
      currMatBeg = 0;
      numrows = numnz = 0;

      for(s=0;s<numScen;s++)   
         for(j=0;j<n;j++)
         {  rmatbeg.push_back(currMatBeg);
            rowname.push_back("q"+to_string(s)+"_"+to_string(j)); numrows++;
            // eps variable
            rmatind.push_back(epsbeg+s*n+j); 
            rmatval.push_back(1.0); 
            numnz++;
            // q variables
            for(i=0;i<m;i++)
            {  index = qbeg + s*n*m +i*n +j;  // int q vars + scenario vars + server offset + client within server
               rmatind.push_back(index); 
               rmatval.push_back(1.0); 
               numnz++;
            }

            sense.push_back('E');
            rhs.push_back(req[s][j]);
            currMatBeg+=m+1;  // m q vars and one eps
         }

      // vector<string> to char**
      char** rname = new char* [rowname.size()];
      for (int index = 0; index < rowname.size(); index++) 
         rname[index] = const_cast<char*>(rowname[index].c_str());

      status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
      delete[] rname;
      if (status)  goto TERMINATE;
   }

   // num assignments x constraints.
   {
      currMatBeg = 0;
      numrows = numnz = 0;
      rmatbeg.clear();
      rowname.clear();
      rmatind.clear();
      rmatval.clear();
      sense.clear();
      rhs.clear();
      for(s=0;s<numScen;s++)
         for(j=0;j<n;j++)
         {  rmatbeg.push_back(currMatBeg);
            rowname.push_back("b"+to_string(s)+"_"+to_string(j)); numrows++;
            for(i=0;i<m;i++)
            {  index = i*n +j;  // x vars, server offset + client within server
               rmatind.push_back(index); 
               rmatval.push_back(1.0); 
               numnz++;
            }
            sense.push_back('L');
            rhs.push_back(b[j]);
            currMatBeg+=m;
         }

      // vector<string> to char**
      char** rname = new char* [rowname.size()];
      for (int index = 0; index < rowname.size(); index++) 
         rname[index] = const_cast<char*>(rowname[index].c_str());

      status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
      delete[] rname;
      if (status)  goto TERMINATE;
   }

   // capacity constraints
   {
      currMatBeg = 0;
      numrows = numnz = 0;
      rmatbeg.clear();
      rowname.clear();
      rmatind.clear();
      rmatval.clear();
      sense.clear();
      rhs.clear();
      for(s=0;s<numScen;s++)
         for(i=0;i<m;i++)
         {  rmatbeg.push_back(currMatBeg);
            rowname.push_back("cap"+to_string(s)+"_"+to_string(i)); numrows++;
            for(j=0;j<n;j++)
            {
               index = qbeg + s*n*m +i*n +j;  // int q vars + scenario vars + server offset + client within server
               rmatind.push_back(index); 
               rmatval.push_back(1.0); 
               numnz++;
            }
            sense.push_back('L');
            rhs.push_back(cap[i]);
            currMatBeg+=n;
         }

      // vector<string> to char**
      char** rname = new char* [rowname.size()];
      for (int index = 0; index < rowname.size(); index++) 
         rname[index] = const_cast<char*>(rowname[index].c_str());

      status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
      delete[] rname;
      if (status)  goto TERMINATE;
   }

   // q-x linking ocnstraints
   {
      currMatBeg = 0;
      numrows = numnz = 0;
      rmatbeg.clear();
      rowname.clear();
      rmatind.clear();
      rmatval.clear();
      sense.clear();
      rhs.clear();
      for(s=0;s<numScen;s++)
         for(i=0;i<m;i++)
            for(j=0;j<n;j++)
            {
               rmatbeg.push_back(currMatBeg);
               rowname.push_back("xq"+to_string(s)+"_"+to_string(i)+"_"+to_string(j)); numrows++;
               index = qbeg + s*n*m +i*n +j;  // int q vars + scenario vars + server offset + client within server
               rmatind.push_back(index); 
               rmatval.push_back(1.0); 
               numnz++;
               index = i*n +j;  // x vars + server offset + client within server
               rmatind.push_back(index); 
               rmatval.push_back(-req[s][j]); 
               numnz++;
               sense.push_back('L');
               rhs.push_back(0);
               currMatBeg+=2;
            }

      // vector<string> to char**
      char** rname = new char* [rowname.size()];
      for (int index = 0; index < rowname.size(); index++) 
         rname[index] = const_cast<char*>(rowname[index].c_str());

      status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
      delete[] rname;
      if (status)  goto TERMINATE;
   }

TERMINATE:
   return (status);
} 

tuple<int,int,int,int,float,float,double,double> HexStochMIP::solveDetEq(int timeLimit, int numScen, bool isVerbose, double epsCost)
{  int i,j,s,epsbeg,qbeg, numcols, solstat, numInfeasibilities = 0;
   double ttot;
   clock_t tstart,tend;
   HxModel model = optimizer.getModel();

   numcols = 0;
   // i: m server, j: n client
   x.resize(n*m);
   for (i = 0; i < n*m; ++i)
   {  x[i] = model.boolVar();
      numcols++;
   }

   // Create columns for eps variables
   epsbeg = numcols; // index of first eps variable
   eps.resize(n*numScen);
   for(i=0;i<numScen;i++)
   {  eps[i] = model.boolVar();
      numcols++;
   }

   // Create the columns for q variables
   qbeg = numcols; // index of first q variabla
   q.resize(numScen*n*m);
   for (i = 0; i < numScen*n*m; ++i)
   {  q[i] = model.intVar(0, maxReq);
      numcols++;
   }


   // ------------------------------------------------------ constraints section

   // quantity q constraints.
   for(s=0;s<numScen;s++)   
      for(j=0;j<n;j++)
      {  HxExpression quant = model.sum();
         for(i=0;i<m;i++)
            quant.addOperand(q[i*n+j]);
         model.constraint(quant==req[j]);
      }

      // assignment
      for(j=0;j<n;j++)
      {  HxExpression nass = model.sum();
         for(i=0;i<m;i++)
            nass.addOperand(x[i*n+j]);
         model.constraint(nass <= b[j]);
      }

      // server capacities
      for(i=0;i<m;i++)
      {  HxExpression usedcap = model.sum();
         for(j=0;j<n;j++)
            usedcap.addOperand(q[i*n+j]);
         model.constraint(usedcap<= cap[i]);
      }

      //link x - q
      for(i=0;i<m;i++)
         for(j=0;j<n;j++)
            model.constraint(q[i*n+j] - req[j]*x[i*n+j] <= 0);

      // -------------------------------------- cost section
      vector<HxExpression> costs(2*m*n); // first x then q
      // Costs due to assignments
      for (int i = 0; i < m; i++)
         //costs[i].resize(n);
         for (int j = 0; j < n; j++)
            costs[i*n + j] = xAssCost[i][j]*x[i*n+j];

      // costs due to infeasibilities
      for (int i = 0; i < m; ++i) 
         ecc ecc

      // rental costs 
      for (int i = 0; i < m; ++i) 
         for (int j = 0; j < n; ++j) 
            costs[n*m + i*n+j] = qcost[i]*q[i*n+j];

      // Minimize the total cost
      totalCost = model.sum(costs.begin(), costs.end());
      model.minimize(totalCost);
      model.close();
      //optimizer.saveEnvironment("lmModel.hxm");

      // --------------------------------------- Parameterize the optimizer
      optimizer.getParam().setTimeLimit(limit);
      tstart = clock();
      optimizer.solve();
      tend = clock();
      ttot = (tend-tstart)/CLOCKS_PER_SEC;
      HxSolutionStatus status = optimizer.getSolution().getStatus();
      cout << "Status " << status << " cost " << totalCost.getValue() << " tcpu " << ttot << endl;
   
ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZz
{
   currMatBeg = 0;
   numrows = numnz = 0;

   for(s=0;s<numScen;s++)   
      for(j=0;j<n;j++)
      {  rmatbeg.push_back(currMatBeg);
   rowname.push_back("q"+to_string(s)+"_"+to_string(j)); numrows++;
   // eps variable
   rmatind.push_back(epsbeg+s*n+j); 
   rmatval.push_back(1.0); 
   numnz++;
   // q variables
   for(i=0;i<m;i++)
   {  index = qbeg + s*n*m +i*n +j;  // int q vars + scenario vars + server offset + client within server
   rmatind.push_back(index); 
   rmatval.push_back(1.0); 
   numnz++;
   }

   sense.push_back('E');
   rhs.push_back(req[s][j]);
   currMatBeg+=m+1;  // m q vars and one eps
      }

   // vector<string> to char**
   char** rname = new char* [rowname.size()];
   for (int index = 0; index < rowname.size(); index++) 
      rname[index] = const_cast<char*>(rowname[index].c_str());

   status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
   delete[] rname;
   if (status)  goto TERMINATE;
}

// num assignments x constraints.
{
   currMatBeg = 0;
   numrows = numnz = 0;
   rmatbeg.clear();
   rowname.clear();
   rmatind.clear();
   rmatval.clear();
   sense.clear();
   rhs.clear();
   for(s=0;s<numScen;s++)
      for(j=0;j<n;j++)
      {  rmatbeg.push_back(currMatBeg);
   rowname.push_back("b"+to_string(s)+"_"+to_string(j)); numrows++;
   for(i=0;i<m;i++)
   {  index = i*n +j;  // x vars, server offset + client within server
   rmatind.push_back(index); 
   rmatval.push_back(1.0); 
   numnz++;
   }
   sense.push_back('L');
   rhs.push_back(b[j]);
   currMatBeg+=m;
      }

   // vector<string> to char**
   char** rname = new char* [rowname.size()];
   for (int index = 0; index < rowname.size(); index++) 
      rname[index] = const_cast<char*>(rowname[index].c_str());

   status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
   delete[] rname;
   if (status)  goto TERMINATE;
}

// capacity constraints
{
   currMatBeg = 0;
   numrows = numnz = 0;
   rmatbeg.clear();
   rowname.clear();
   rmatind.clear();
   rmatval.clear();
   sense.clear();
   rhs.clear();
   for(s=0;s<numScen;s++)
      for(i=0;i<m;i++)
      {  rmatbeg.push_back(currMatBeg);
   rowname.push_back("cap"+to_string(s)+"_"+to_string(i)); numrows++;
   for(j=0;j<n;j++)
   {
      index = qbeg + s*n*m +i*n +j;  // int q vars + scenario vars + server offset + client within server
      rmatind.push_back(index); 
      rmatval.push_back(1.0); 
      numnz++;
   }
   sense.push_back('L');
   rhs.push_back(cap[i]);
   currMatBeg+=n;
      }

   // vector<string> to char**
   char** rname = new char* [rowname.size()];
   for (int index = 0; index < rowname.size(); index++) 
      rname[index] = const_cast<char*>(rowname[index].c_str());

   status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
   delete[] rname;
   if (status)  goto TERMINATE;
}

// q-x linking ocnstraints
{
   currMatBeg = 0;
   numrows = numnz = 0;
   rmatbeg.clear();
   rowname.clear();
   rmatind.clear();
   rmatval.clear();
   sense.clear();
   rhs.clear();
   for(s=0;s<numScen;s++)
      for(i=0;i<m;i++)
         for(j=0;j<n;j++)
         {
            rmatbeg.push_back(currMatBeg);
            rowname.push_back("xq"+to_string(s)+"_"+to_string(i)+"_"+to_string(j)); numrows++;
            index = qbeg + s*n*m +i*n +j;  // int q vars + scenario vars + server offset + client within server
            rmatind.push_back(index); 
            rmatval.push_back(1.0); 
            numnz++;
            index = i*n +j;  // x vars + server offset + client within server
            rmatind.push_back(index); 
            rmatval.push_back(-req[s][j]); 
            numnz++;
            sense.push_back('L');
            rhs.push_back(0);
            currMatBeg+=2;
         }

   // vector<string> to char**
   char** rname = new char* [rowname.size()];
   for (int index = 0; index < rowname.size(); index++) 
      rname[index] = const_cast<char*>(rowname[index].c_str());

   status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
   delete[] rname;
   if (status)  goto TERMINATE;

   // Open output file in append mode (std::ios::app)

   if(isVerbose || true)
   {  status = CPXgetslack(env,lp,&slack[0],0,cur_numrows-1);
      if (status)
      { cout<<"Failed to get optimal slack values."<<endl; goto TERMINATE; }

      //for (i = 0; i < cur_numrows; i++) 
      //   cout << "Row " << i << ":  Slack = " << slack[i] << endl;
      //
      for (j = 0; j<cur_numcols; j++)
         if (x[j]>0.01)
         {
            cout<<"Column "<<j<<":  Value = "<<x[j]<<endl;
            outFile<<"Column "<<j<<":  Value = "<<x[j]<<endl;
         }
   }

   // eps variables, infeasibilities
   for(j=n*m;j<(n*m+n*numScen);j++)
      if (x[j]>0.01)
      {  cout<<"Eps "<<j<<":  Value = "<<x[j]<<endl;
         outFile<<"Eps "<<j<<":  Value = "<<x[j]<<endl;
         numInfeasibilities++;
      }
   cout << "Number of infeasibilities: " << numInfeasibilities << endl;
   outFile.close();

   }

   // Free up the CPLEX environment, if necessary
   if (env != NULL) 
   {  status = CPXcloseCPLEX(&env);
      if (status) 
      {  char  errmsg[CPXMESSAGEBUFSIZE];
         cout << "Could not close CPLEX environment." << endl;
         CPXgeterrorstring(env, status, errmsg);
         cout << errmsg << endl;
      }
   }

   tuple<int,int,int,int,float,float,double,double> res = make_tuple(status,cur_numcols,cur_numrows,numInfeasibilities,zlb,objval,lbfinal,total_time);
   return res;
}  /* END main */

// random quantity in the boosted series according to empyrical distrib
int HexStochMIP::generateReq(int j, int nboost)
{  int ind,val;
   ind = rand()%nboost;
   val = boostFcasts[j][ind]; // ind-th element of the series for j-th client
   return val;
}