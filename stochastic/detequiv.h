#ifndef DETEQ_H
#define DETEQ_H

#include "common.h"

class StochMIP
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
   vector<vector<int>> boostFcasts;

   void readInstance(string& fileName);
   int readBoostForecasts(string filePath,int nboost);
};
#endif // DETEQ_H