#include "common.h"

class ETSDetequiv : public Base
{
public:
   string name;

   int n,m;    // num clients e servers
   int nmult, maxReq;

   vector<int> b;       // number of allowed assignments
   vector<int> cap;     // DC capacities
   vector<int> req;     // client requests
   vector<int> cols;    // cost matrix columns
   vector<int> rows;    // cost matrix rows
   vector<int> qcost;   // DC rental cost

   vector<vector<int>> xAssCost; // assignment costs

   // Hexaly Optimizer
   HexalyOptimizer optimizer;

   // Decision variables
   vector<HxExpression> x;
   vector<HxExpression> q;

   // Objective
   HxExpression totalCost;

   // Read instance data
   void readInstance(string& fileName) 
   {  string line;
      int i,j;

      ifstream infile;
      cout << "Opening " << fileName << endl;
      infile.exceptions(ifstream::failbit | ifstream::badbit);
      infile.open(fileName.c_str());

      std::stringstream buffer;
      buffer << infile.rdbuf();
      line = buffer.str();
      //std::getline(jData,line);
      infile.close();

      json::Value JSV = json::Deserialize(line);
      name = JSV["name"];
      n = (int)JSV["n"];
      m = (int)JSV["m"];
      nmult = (int)JSV["nmult"];

      cap.resize(m,0);
      for (i = 0; i < JSV["cap"].size(); i++)
         cap[i] = JSV["cap"][i];

      maxReq = 0;
      req.resize(n,0);
      for (i = 0; i < JSV["req"].size(); i++)
      {  req[i] = JSV["req"][i];
         if(req[i]>maxReq) maxReq = req[i];
      }

      cols.resize(n);
      for (i = 0; i < JSV["cols"].size(); i++)
         cols[i] = JSV["cols"][i];

      rows.resize(m);
      for (i = 0; i < JSV["rows"].size(); i++)
         rows[i] = JSV["rows"][i];

      b.resize(n);
      for (i = 0; i < JSV["b"].size(); i++)
         b[i] = JSV["b"][i];

      qcost.resize(m);
      for (i = 0; i < JSV["qcost"].size(); i++)
         qcost[i] = JSV["qcost"][i];


      fileName = "c:/git/GDOforecast/generator/seedMatrix.csv";
      cout << "Opening " << fileName << endl;
      infile.exceptions(ifstream::failbit | ifstream::badbit);
      infile.open(fileName);
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
   }

   // Declare the optimization model
   void solve(int limit) 
   {  int i,j;
      double ttot;
      clock_t tstart,tend;
      HxModel model = optimizer.getModel();

      // i: server, j: client
      x.resize(n*m);
      for (i = 0; i < n*m; ++i)
         x[i] = model.boolVar();

      q.resize(n*m);
      for (i = 0; i < n*m; ++i)
         q[i] = model.intVar(0,maxReq);

      // client requests constraints
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

      vector<HxExpression> costs(2*m*n); // first x then q
      // Costs due to assignments
      for (int i = 0; i < m; i++)
         //costs[i].resize(n);
         for (int j = 0; j < n; j++)
            costs[i*n + j] = xAssCost[i][j]*x[i*n+j];

      // rental costs 
      for (int i = 0; i < m; ++i) 
         for (int j = 0; j < n; ++j) 
            costs[n*m + i*n+j] = qcost[i]*q[i*n+j];

      // Minimize the total cost
      totalCost = model.sum(costs.begin(), costs.end());
      model.minimize(totalCost);
      model.close();
      //optimizer.saveEnvironment("lmModel.hxm");

      // Parameterize the optimizer
      optimizer.getParam().setTimeLimit(limit);
      tstart = clock();
      optimizer.solve();
      tend = clock();
      ttot = (tend-tstart)/CLOCKS_PER_SEC;
      HxSolutionStatus status = optimizer.getSolution().getStatus();
      cout << "Status " << status << " cost " << totalCost.getValue() << " tcpu " << ttot << endl;
   }

   // Write the solution in a file with the following format:
   // - value of the objective
   // - indices of the facilities (between 0 and N-1)
   void writeSolution(const string& fileName) 
   {  ofstream outfile;
      outfile.exceptions(ofstream::failbit | ofstream::badbit);
      outfile.open(fileName.c_str());

      outfile << totalCost.getValue() << endl;
      for (int i = 0; i < n; ++i) 
         if (x[i].getValue() == 1)
            outfile << i << " ";
      outfile << endl;
   }
};
