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
   vector<int> cols;    // cost matrix columns
   vector<int> rows;    // cost matrix rows
   vector<int> qcost;   // DC rental cost
   vector<double> p;    // probability of the scenarios

   vector<vector<int>> req;      // client requests, scenario / client
   vector<vector<int>> xAssCost; // assignment costs
   vector<vector<int>> boostFcasts;

   void readInstance(string& fileName, int numScen, int nboost, int nmult);
   int readBoostForecasts(string filePath,int nboost,int numScen);
   int populateTableau(CPXENVptr env, CPXLPptr lp, int numScen, double epsCost);
   tuple<int,int,int,int,float,float,double,double> solveDetEq(int timeLimit, int numScen, bool isVerbose, double epsCost);
   int generateReq(int j, int nboost);
};
#endif // DETEQ_H