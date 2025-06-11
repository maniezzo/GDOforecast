#pragma once
#include "c:/hexaly_13_5/include/optimizer/hexalyoptimizer.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <time.h>
#include "json.h"

using namespace hexaly;
using namespace std;

class Base {
public:
   virtual void readInstance(string& fileName) = 0;
   virtual void solve(int limit) = 0;
   virtual void writeSolution(const string& fileName) = 0;
   virtual ~Base() = default;
};
