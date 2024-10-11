#include "detequiv.h"

// Read instance data, generates also the p and the req
void StochMIP::readInstance(string& fileName, int numScen, int nboost, int nmult) 
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
   // make splittable the nmult clients with higer requests
   for(i=0;i<nmult;i++)
      b[ind[i]] = 2;

   qcost.resize(m);
   for (i = 0; i < JSV["qcost"].size(); i++)
      qcost[i] = JSV["qcost"][i];

   string matrixFile = "../generator/seedMatrix.csv";
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
int StochMIP::readBoostForecasts(string filePath,int nboost,int numScen)
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
      {  req[s][j] = generateReq(j,nboost);
         if(req[s][j]>maxReq) maxReq = req[s][j];
      }
   }

   return 0;
}

// The tableu for the bscenario case.
int StochMIP::populateTableau(CPXENVptr env, CPXLPptr lp, int numScen, double epsCost)
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

tuple<int,int,int,int,float,float,double> StochMIP::solveDetEq(int timeLimit, int numScen, bool isVerbose, double epsCost)
{  int      solstat, numInfeasibilities = 0;
   double   objval,zlb;
   vector<double> x;
   vector<double> pi;
   vector<double> slack;
   vector<double> dj;
   vector<char>   ctype;

   CPXENVptr  env = NULL;
   CPXLPptr  lp  = NULL;
   int       status = 0;
   int       i,j,s;
   int       cur_numrows, cur_numcols;
   time_t    tstart, tend;
   double    total_time;

   // Initialize the CPLEX environment
   env = CPXopenCPLEX(&status);
   if (env == NULL) 
   {  char  errmsg[CPXMESSAGEBUFSIZE];
      cout << "Could not open CPLEX environment." << endl;
      CPXgeterrorstring(env, status, errmsg);
      cout << errmsg << endl;
      goto TERMINATE;
   }

   // Turn on output to the screen 
   status = CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_ON);
   if (status) 
   {  cout << "Failure to turn on screen indicator, error " << status << endl; goto TERMINATE; }

   // Turn on data checking
   status = CPXsetintparam(env, CPXPARAM_Read_DataCheck,CPX_DATACHECK_WARN);
   if (status) 
   {  cout << "Failure to turn on data checking, error " << status << endl; goto TERMINATE; }

   // time limit
   status = CPXsetdblparam(env, CPX_PARAM_TILIM, timeLimit);

   // Create the problem.
   lp = CPXcreateprob(env, &status, "scenGDO");
   if (lp == NULL) 
   {  cout << "Failed to create LP." << endl; goto TERMINATE; }

   // Now populate the problem with the data.
   status = populateTableau(env, lp, numScen, epsCost);
   if (status) 
   {  cout <<"Failed to populate problem." << endl; goto TERMINATE; }

   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);
   cout << "LP model; ncol=" << cur_numcols << " nrows=" << cur_numrows << endl;

   // Finally, write a copy of the problem to a file
   if(cur_numcols < 1000)
   {  status = CPXwriteprob(env,lp,"problem.lp",NULL);
      if (status) 
      {  cout << "Failed to write model to disk." << endl;
         goto TERMINATE;
      }
   }

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   status = CPXlpopt(env, lp);
   if (status) 
   {  cout << "Failed to optimize LP." << endl; goto TERMINATE; }

   // save solutions
   for(int j=0;j<cur_numcols;j++)
   {  x.push_back(0);   // primal variables
      dj.push_back(0);  // reduced costs
   }

   for (int i = 0; i < cur_numrows; i++)
   {  pi.push_back(0);     // dual variables
      slack.push_back(0);  // constraint slacks
   }

   status = CPXsolution(env, lp, &solstat, &objval, &x[0], &pi[0], &slack[0], &dj[0]);
   if (status) 
   {  cout << "Failed to obtain solution." << endl; goto TERMINATE; }

   zlb = objval;

   // Write the output to the screen.
   cout << "\nSolution status = " << solstat << endl;
   cout << "Solution value  = "   << objval << endl;

   if(isVerbose)
   {  //for (i = 0; i < cur_numrows; i++) 
      //   cout << "Row "<< i << ":  Slack = "<< slack[i] <<"  Pi = " << pi[i] << endl;

      for (j = 0; j<cur_numcols; j++)
         if (x[j]>0.01)
            cout<<"Column "<<j<<":  Value = "<<x[j]<<"  Reduced cost = "<<dj[j]<<endl;
   }

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MIP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

   // Now copy the ctype array
   for(s=0;s<numScen;s++)
      for(i=0;i<m;i++)
         for(j=0;j<n;j++)
            ctype.push_back('I');   // x vars
   for(s=0;s<numScen;s++)
      for(j=0;j<n;j++)
         ctype.push_back('I');      // eps vars
   for(s=0;s<numScen;s++)
      for(i=0;i<m;i++)
         for(j=0;j<n;j++)
            ctype.push_back('I');   // q vars
   status = CPXcopyctype(env, lp, &ctype[0]);
   if (status)
   {  cout << "Failed to copy ctype" << endl; goto TERMINATE; }

   // -------------------------------------------------- Optimize to integrality
   tstart = clock();
   status = CPXmipopt(env, lp);
   if (status) 
   {  cout << "Failed to optimize MIP" << endl; goto TERMINATE; }
   tend = clock();
   total_time = (double)( tend - tstart )/(double)CLK_TCK ;
   cout << "Elapsed time :" << total_time << endl;

   solstat = CPXgetstat(env, lp);
   cout << "Solution status = " << solstat << endl; // 101 CPXMIP_OPTIMAL,  102 CPXMIP_OPTIMAL_TOL (opt within tolerance)

   status = CPXgetobjval(env, lp, &objval);
   if (status) 
   {  cout << "No MIP objective value available.  Exiting..." << endl; goto TERMINATE; }

   cout << "Solution value  = " << objval << endl;
   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);
   status = CPXgetx(env,lp,&x[0],0,cur_numcols-1);
   if (status)
   { cout<<"Failed to get optimal integer x."<<endl; goto TERMINATE; }

   if(isVerbose)
   {  status = CPXgetslack(env,lp,&slack[0],0,cur_numrows-1);
      if (status)
      { cout<<"Failed to get optimal slack values."<<endl; goto TERMINATE; }

      //for (i = 0; i < cur_numrows; i++) 
      //   cout << "Row " << i << ":  Slack = " << slack[i] << endl;
      //
      //for (j = 0; j<cur_numcols; j++)
      //   if (x[j]>0.01)
      //      cout<<"Column "<<j<<":  Value = "<<x[j]<<endl;
   }

   // eps variables, infeasibilities
   for(j=n*m;j<(n*m+n*numScen);j++)
      if (x[j]>0.01)
      {  cout<<"Eps "<<j<<":  Value = "<<x[j]<<endl;
         numInfeasibilities++;
      }
   cout << "Number of infeasibilities: " << numInfeasibilities << endl;

TERMINATE:
   // Free up the problem as allocated by CPXcreateprob, if necessary
   if (lp != NULL) 
   {  status = CPXfreeprob(env, &lp);
      if (status) 
         cout << "CPXfreeprob failed, error code " << status << endl;
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

   tuple<int,int,int,int,float,float,double> res = make_tuple(status,cur_numcols,cur_numrows,numInfeasibilities,zlb,objval,total_time);
   return res;
}  /* END main */

// random quantity in the boosted series according to empyrical distrib
int StochMIP::generateReq(int j, int nboost)
{  int ind,val;
   ind = rand()%nboost;
   val = boostFcasts[j][ind]; // ind-th element of the series for j-th client
   return val;
}