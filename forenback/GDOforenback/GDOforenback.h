#pragma once
#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>   // stringstream
#include <iomanip>
#include <limits>
#include <climits>
#include <cmath>
#include <string>
#include <vector>
#include <array>
#include <list>
#include <cstdlib>    /* srand, rand */
#include <time.h>     /* time */
#include <assert.h>   /* assert */
#include <algorithm>  /* std::sort, std::copy */
#include "json.h"

using namespace std;

int m,n,zub,maxNodes,maxIter,delta;
bool isVerbose;
vector<int> cap, capleft;
vector<int> sol,solbest;
vector<vector<int>> c;
vector<vector<int>> req;

// the state of each partial solution
struct node
{
   int z;      // cost of the partial soluton
   int client; // the assigned client
   int server; // who the client is assigned to
   int dad;    // which sol was expanded into this
   vector<int> capused;  // array of used capacities
};

vector<node> stack;   // stack all nodes expanded during search
vector<list<int>> fTree;   // forward tree, one list for each level / customer
vector<list<int>> bTree;   // backward tree, one list for each level / customer
vector<list<int>> fList;   // the list of still unexpanded nodes at each level of the forward tree
vector<list<int>> bList;   // the list of still unexpanded nodes at each level of the backward tree
vector<int> fTopCost; // max cost of expanded node at each level of the forward tree
vector<int> bTopCost; // max cost of expanded node at each level of the backward tree

int indLastNode;  // aka stack.size(). just it
int numFathomed;  // num fahtomed nodes
int expandNode(ofstream& flog, int iter, vector<vector<int>> c, int j, int jlev, int currNode, vector<int> indCost, bool isForward);
int readSolutionF(ofstream& flog, int currNode, vector<int> indCost);
int insertInOrder(list<int>& lst, int elem);
int read_data();
int checkSol(vector<int> sol);
int sweepForward(ofstream& flog, int iter, vector<vector<int>> c, int delta, int maxNodes, int openNodes, vector<int> indCost);
int findNextNodeF(int jlev, int newNodes, int openNodes);