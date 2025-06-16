#include "HexESdetequiv.h"

// deterministic equivalent, Exponential Smoothing (Holt Winters) case, Hexaly solver

// Read instance data, generates also the p and the req
void HexStochMIP::readInstance(string& fileName, int numScen, int nboost, int nmult) 
{  string line;
   int i,j;

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
   boostFcasts.resize(nboost);
   for(i=0;i<nboost;i++)
      boostFcasts[i].resize(n);

   i=0;
   while (i<numScen && getline(infile, line))
   {  stringstream ss(line);
      string value;

      // Split the line by commas
      j=0;
      while (getline(ss, value, ',') && j<n) 
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
      req[s].resize(n);
   for(s=0;s<numScen;s++)
      for(j=0;j<n;j++)
      {  req[s][j] = boostFcasts[s][j]; //generateReq(j,nboost);
         if(req[s][j]>maxReq) maxReq = req[s][j];
      }

   return 0;
}


// confronta la soluzione robusta con i dati veri futuri
void HexStochMIP::checkSolution(string histFile)
{  int i,j;
   string line;
   vector<int> histData;
   vector<int> assignments(52); // vettore per le assegnazioni dei clienti

   // legge serie storiche di dicembre
   cout<<"opening historical data file: "<<histFile<<endl;
   ifstream infile(histFile);
   if (!infile.is_open())
   {  cout<<"Error opening historical data file: "<<histFile<<endl;
      return;
   }
   getline(infile, line); // headers

   i=0;
   while (getline(infile, line))
   {  stringstream ss(line);
      string value;

      // valore storico efettivo di dicembre
      j=0;
      while (getline(ss, value, ','))
      {  if ((i+1)%48 == 0)
         {  //cout<<(i+1)/48<<"_"<<j<<") "<<value<<endl; // = round(stof(value)); 
            if(j==3)
               histData.push_back(round(stof(value)));
         }
         j++;
      }
      i++;
   }      
   infile.close();

   // il file "hexaly_results.dat" e' riempito in HexESdetequiv.cpp
   cout<<"opening hexaly_results.dat"<<endl;
   ifstream infile2("hexaly_results.dat");
   if (!infile2.is_open())
   {  cout<<"Error opening detequiv data file: "<<histFile<<endl;
      return;
   }
   while (getline(infile2, line))
   {  stringstream ss(line);
      string value;

      // variabili in soluzione robusta
      j=0;
      while (getline(ss, value, ' '))
      { 
         if(j==1)
            i=round(stof(value));
         if(j==2)
            assignments[round(stof(value))] = i;
         j++;
      }
   }
   infile2.close();

   // controlla ammisibilità delle capacità
   int totCost = 0; // costo totale della soluzione
   vector<int> whCost(m);  // costi di ogni magazzino
   vector<int> capUsed(m); // vettore per le capacità usate
   for(j=0;j<n;j++)
   {
      capUsed[assignments[j]] += histData[j]; // somma le richieste dei clienti assegnati al server
      totCost += xAssCost[assignments[j]][j]; // somma i costi di assegnamento
      totCost += qcost[assignments[j]]*histData[j]; // somma i costi di noleggio
      whCost[assignments[j]] += qcost[assignments[j]]*histData[j]; // costo di noleggio per il magazzino
   }
   for (i=0;i<m;i++)
   {
      cout<<"Server "<<i<<" cost "<< whCost[i] <<" used capacity "<<capUsed[i]<<" / "<<cap[i]<<endl;
      if (capUsed[i]>cap[i])
         cout<<">>>>>>> Server "<<i<<" capacity exceeded: "<<capUsed[i]<<" > "<<cap[i]<<endl;
   }
}


tuple<int,int,int,int,float,float,double,double> HexStochMIP::solveDetEq(int timeLimit, int numScen, bool isVerbose, double epsCost, string histFile)
{  int i,j,s,k,epsbeg,qbeg,maxCap, numcols, numrows, totcost, numInfeasibilities = 0;
   double ttot;
   clock_t tstart,tend;
   HxModel model = optimizer.getModel();

   numcols = numrows = 0;
   // i: m server, j: n client
   x.resize(n*m);
   for (k = 0; k < n*m; ++k)
   {  x[k] = model.boolVar();
      numcols++;
   }

   // Create columns for eps variables
   epsbeg = numcols; // index of first eps variable
   eps.resize(n*numScen);
   for(k=0;k<numScen*n;k++)
   {  eps[k] = model.intVar(0, maxReq);
      numcols++;
   }

   // Create the columns for q variables
   qbeg = numcols; // index of first q variabla
   q.resize(numScen*n*m);
   maxCap = 0;
   for(i=0;i<m;i++)
      if (cap[i]>maxCap) maxCap = cap[i]; // max capacity of the servers
   // order: i - j - s
   for (k = 0; k < numScen*n*m; ++k)
   {  q[k] = model.intVar(0, maxCap);
      numcols++;
   }

   // -------------------------------------- cost section: x - eps - q
   vector<HxExpression> costs(m*n + n*numScen + n*m*numScen); 
   // Costs due to assignments
   for (i = 0; i < m; i++)
      //costs[i].resize(n);
      for (j = 0; j < n; j++)
         costs[i*n + j] = xAssCost[i][j]*x[i*n+j]; // costo assegnamento x

   // costs due to infeasibilities
   for (j = 0; j < n; ++j)
      for(s=0;s< numScen; ++s) 
         costs[n*m+j*numScen+s] = epsCost*eps[j*numScen+s]; // costo eps, per scenario e cliente

   // rental costs 
   for (i = 0; i < m; ++i) 
      for (j = 0; j < n; ++j) 
         for(s=0;s<numScen;++s) 
            costs[n*m + n*numScen + i*n*numScen + j*numScen + s] = qcost[i]*q[i*n*numScen + j*numScen + s]; // costo noleggio q, per scenario e cliente

   // ------------------------------------------------------ constraints section
   
   // quantity q constraints.
   for(s=0;s<numScen;s++)   
      for(j=0;j<n;j++)
      {  HxExpression quant = model.sum();
         for(i=0;i<m;i++)
            quant.addOperand(q[i*n*numScen+j*numScen+s]);
         quant.addOperand(eps[j*numScen+s]);
         model.constraint(quant==req[s][j]);
         numrows++;
      }

   // server capacities
   for(i=0;i<m;i++)
      for(s=0;s<numScen;s++)   
      {  HxExpression usedcap = model.sum();
         for(j=0;j<n;j++)
            usedcap.addOperand(q[i*n*numScen+j*numScen+s]);
         model.constraint(usedcap <= cap[i]);
         numrows++;
      }

   // assignment
   for(j=0;j<n;j++)
   {  HxExpression nass = model.sum();
      for(i=0;i<m;i++)
         nass.addOperand(x[i*n+j]);
      model.constraint(nass <= b[j]);
      numrows++;
   }

   //link x - q
   for(s=0;s<numScen;s++)   
      for(i=0;i<m;i++)
         for(j=0;j<n;j++)
         {  model.constraint(q[i*n*numScen+j*numScen+s]-req[s][j]*x[i*n+j]<=0);
            numrows++;
         }

   // Minimize the total cost
   totalCost = model.sum(costs.begin(), costs.end());
   model.minimize(totalCost);
   model.close();
   optimizer.saveEnvironment("lmModel.hxm");
   cout << "n. var."<<numcols<<" num.x "<<epsbeg<<" num eps "<<qbeg-epsbeg<<endl;

   // --------------------------------------- Parametrize the optimizer
   optimizer.getParam().setTimeLimit(timeLimit);
   tstart = clock();
   optimizer.solve();
   tend = clock();
   ttot = (tend-tstart)/CLOCKS_PER_SEC;
   HxSolutionStatus status = optimizer.getSolution().getStatus();
   if (optimizer.getSolution().getStatus() == SS_Inconsistent)    cout << "Inconsistent model" << endl;
   else if (optimizer.getSolution().getStatus() == SS_Infeasible) cout << "Infeasible model" << endl;
   else if (optimizer.getSolution().getStatus() == SS_Feasible)   cout << "Feasible solution found" << endl;
   else if (optimizer.getSolution().getStatus() == SS_Optimal)    cout << "Optimal solution found" << endl;
   else cout<<"Unknown status"<<endl;

   // Open output file in append mode (std::ios::app)
   ofstream outFile;
   outFile.exceptions(ofstream::failbit | ofstream::badbit);
   outFile.open("hexaly_results.dat");

   try
   {  cout << "Status " << status << " cost " << totalCost.getDoubleValue() << " tcpu " << ttot << endl;
   }
   catch(int errorCode)
   {  cout<<"errorcode "<<errorCode<<endl; }

   if(isVerbose || true)
   {  totcost = 0;
      for (i = 0; i < m; i++)
         for (j = 0; j < n; j++)
         {  k = i*n+j;
            if (x[k].getValue()>0.01)
            {  cout<<"i "<<i<<" j "<<j<<" x["<<k<<"]:  Value = "<<x[k].getValue()<<endl;
               outFile<<"x " << i << " " << j << " k="<<k<<":  Val. = "<<x[k].getValue()<< " cost " << xAssCost[i][j] << endl;
               totcost += xAssCost[i][j]*x[k].getValue();
            }
         }
   }
   cout<<"Total assignment cost: "<<totcost<<endl;

   // eps variables, infeasibilities
   for(k=0;k<eps.size();k++)
      if (eps[k].getValue() > 0.01)
      {  cout<<"Eps "<<k<<":  Value = "<<eps[k].getValue()<<endl;
         outFile<<"Eps "<<k<<":  Value = "<<eps[k].getValue()<<endl;
         numInfeasibilities++;
      }
   cout << "Number of infeasibilities: " << numInfeasibilities << endl;
   outFile.close();

   checkSolution(histFile);

   tuple<int,int,int,int,float,float,double,double> res = make_tuple(status,numcols,numrows,numInfeasibilities,-1,totalCost.getDoubleValue(),-1,ttot);
   return res;
}  /* END main */

// random quantity in the boosted series according to empyrical distrib
int HexStochMIP::generateReq(int j, int nboost)
{  int ind,val;
   ind = rand()%nboost;
   val = boostFcasts[j][ind]; // ind-th element of the series for j-th client
   return val;
}