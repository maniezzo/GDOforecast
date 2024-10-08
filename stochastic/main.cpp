#include "stochastic.h"
#include "detequiv.h"
#include "json.h"

int main()
{  StochMIP Stoch;
   SingleMIP MIP;
   string instanceFile,distribFile,solFile;
   string line;
   int status;
   //srand(666);
   srand(time(NULL));

   ifstream infile;
   cout<<"Opening config.json"<<endl;
   infile.exceptions(ifstream::failbit | ifstream::badbit);
   infile.open("config.json");

   std::stringstream buffer;
   buffer << infile.rdbuf();
   line = buffer.str();
   infile.close();
   json::Value JSV = json::Deserialize(line);

   instanceFile   = JSV["instanceFile"];
   distribFile    = JSV["distribFile"];
   solFile        = JSV["solFile"];
   int numScen    = JSV["numScen"];      // numero d iscenari da generare
   int TimeLimit  = JSV["TimeLimit"];    // CPLEX time limit
   double epsCost = JSV["epsCost"];      // costo ogni infeasibility
   bool isVerbose = JSV["isVerbose"];
   bool isDetmnst = JSV["isDetmnst"];    // run also the deterministic version

   // Find the last occurrence of '_' and '.' and extract number of boosted series
   size_t underscorePos = distribFile.rfind('_');
   size_t dotPos        = distribFile.rfind('.');
   string strNboost = distribFile.substr(underscorePos + 1, dotPos - underscorePos - 1);
   int nboost = stoi(strNboost);

   // non stochastic, just a test
   if(isDetmnst)
   {  MIP.readInstance(instanceFile);
      status = MIP.solveMIP(TimeLimit);
   }

   // stochastic, deterministic equivalent
   Stoch.readInstance(instanceFile, numScen,nboost);
   Stoch.readBoostForecasts(distribFile,nboost,numScen);
   tuple<int,int,int,int,float,double> res = Stoch.solveDetEq(TimeLimit,numScen,isVerbose, epsCost);
   size_t slashPos = instanceFile.rfind('/');
   dotPos = instanceFile.rfind('.');
   string strInst = instanceFile.substr(slashPos + 1, dotPos - slashPos - 1);
   ostringstream osString;
   osString << "Instance "     << strInst;
   osString << " num.scen. "   << numScen;
   osString << " num.boost "   << nboost;
   osString << " status "      << get<0>(res);
   osString << " cur_numcols " << get<1>(res);
   osString << " cur_numrows " << get<2>(res);
   osString << " numInfeasibilities " << get<3>(res);
   osString << " objval "      << get<4>(res);
   osString << " total_time "  << get<5>(res) << endl;
   string outStr = osString.str();
   cout << outStr << endl;

   // Open the file in append mode (std::ios::app)
   std::ofstream outFile(solFile, std::ios::app);
   if (!outFile) 
   {  cout << "Error opening output file: " << solFile << std::endl;
   }
   else
   {  outFile<<outStr;
      outFile.close();
   }
   cout << "<ENTER> to exit ..."; getchar();
}
