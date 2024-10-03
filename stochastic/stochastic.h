#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <ilcplex/cplex.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <time.h>
#include "json.h"

using namespace std;

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
