#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include "common.h"

class SingleMIP
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

   int populateTableau(CPXENVptr env, CPXLPptr lp);
   tuple<int,int,int,float,float,double,double> solveMIP(int timeLimit, bool isVerbose);
   void readInstance(string& fileName);
   tuple<vector<int>,vector<int>> generateQcosts();

};
#endif // STOCHASTIC_H