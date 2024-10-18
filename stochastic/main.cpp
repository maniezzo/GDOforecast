#include "deterministic.h"
#include "detequiv.h"
#include "json.h"

// New callback function for CPLEX 22.x
int CPXPUBLIC myCallbackFunction(CPXCALLBACKCONTEXTptr context, CPXLONG contextid, void *cbhandle) {
   // Cast the callback handle to our custom data structure
   CallbackData *data = static_cast<CallbackData*>(cbhandle);

   // Get the current time
   auto currentTime = std::chrono::steady_clock::now();

   // Calculate time elapsed since the last print
   std::chrono::duration<double> elapsedTime = currentTime - data->lastPrintTime;

   // Check if 10 minutes (600 seconds) have passed
   if (elapsedTime.count() >= 600.0) {
      double bestObjVal, incumbObjVal;

      // Check the context and retrieve bounds accordingly
      if (contextid == CPX_CALLBACKCONTEXT_RELAXATION) {
         // Get the dual bound (upper bound) from the relaxation context
         if (CPXcallbackgetinfodbl(context, CPX_CALLBACK_INFO_DUAL_BOUND, &bestObjVal)) {
            std::cerr << "Error retrieving dual bound (upper bound) in relaxation." << std::endl;
            return 1;
         }
      } 
      else if (contextid == CPX_CALLBACKCONTEXT_CANDIDATE) {
         // Get the incumbent value (lower bound) from the candidate context
         if (CPXcallbackgetinfodbl(context, CPX_CALLBACK_INFO_BEST_SOL, &incumbObjVal)) {
            std::cerr << "Error retrieving incumbent value (lower bound) in candidate." << std::endl;
            return 1;
         }
      }

      // Print the bounds (if both were retrieved)
      std::cout << "After " << elapsedTime.count() / 60 << " minutes: "
         << "Best Upper Bound (Dual): " << bestObjVal
         << ", Best Lower Bound (Incumbent): " << incumbObjVal << std::endl;

      // Reset the timer
      data->lastPrintTime = currentTime;
   }

   return 0; // Zero return indicates success
}

int main()
{  StochMIP Stoch;
   SingleMIP MIP;
   string instanceFile,distribFile,solFile;
   string line;
   stringstream ss;
   int status,irep,nrep,nmult;
   float zlb;
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
   nrep  = JSV["nrep"];                  // number of repetitions
   nmult = JSV["nmult"];                 // number of split clients, with b[i]=2

   // Find the last occurrence of '_' and '.' and extract number of boosted series
   size_t underscorePos = distribFile.rfind('_');
   size_t dotPos        = distribFile.rfind('.');
   string strNboost = distribFile.substr(underscorePos + 1, dotPos - underscorePos - 1);
   int nboost = stoi(strNboost);

   // non stochastic, no boosting
   if(isDetmnst)
   {  
      for(irep=0;irep<nrep;irep++)
      {  ss.str("");
         ss << irep;
         string inst = instanceFile+"_"+ss.str()+".json";
         MIP.readInstance(inst);
         tuple<int,int,int,float,float,double,double> res = MIP.solveMIP(TimeLimit,isVerbose);

         size_t slashPos = inst.rfind('/');
         dotPos = inst.rfind('.');
         string strInst = inst.substr(slashPos+1,dotPos-slashPos-1);
         ostringstream osString;
         osString<<"Instance "<<strInst;
         osString<<" num.scen. "<<numScen;
         osString<<" num.boost " << nboost;
         osString<<" status " <<    get<0>(res);
         osString<<" cur_numcols "<<get<1>(res);
         osString<<" cur_numrows "<<get<2>(res);
         osString<<" zlb "<<        get<3>(res);
         osString<<" objval "<<     get<4>(res);
         osString<<" finalLb "<<    get<5>(res);
         osString<<" total_time "<< get<6>(res)<<endl;
         string outStr = osString.str();
         cout<< fixed << outStr<<endl;

         // Open the file in append mode (std::ios::app)
         std::ofstream outFile(solFile,std::ios::app);
         if (!outFile)
            cout<<"Error opening output file: "<<solFile<<std::endl;
         else
         {  outFile<<outStr;
            outFile.close();
         }
      }
   }
   else
   {  // stochastic, deterministic equivalent
      instanceFile += ".json";
      for(irep=0;irep<nrep;irep++)
      {
         Stoch.readInstance(instanceFile,numScen,nboost,nmult);
         Stoch.readBoostForecasts(distribFile,nboost,numScen);
         tuple<int,int,int,int,float,float,double,double> res = Stoch.solveDetEq(TimeLimit,numScen,isVerbose,epsCost);
         size_t slashPos = instanceFile.rfind('/');
         dotPos = instanceFile.rfind('.');
         string strInst = instanceFile.substr(slashPos+1,dotPos-slashPos-1);
         ostringstream osString;
         osString<<"Instance "<<strInst;
         osString<<" num.scen. "<<numScen;
         osString<<" num.boost " << nboost;
         osString<<" repet. " << irep;
         osString<<" status " << get<0>(res);
         osString<<" cur_numcols "<<get<1>(res);
         osString<<" cur_numrows "<<get<2>(res);
         osString<<" numInfeasibilities "<<get<3>(res);
         osString<<" zlb "<<get<4>(res);
         osString<<" objval "<<get<5>(res);
         osString<<" finalLb "<<get<6>(res);
         osString<<" total_time "<<get<7>(res)<<endl;
         string outStr = osString.str();
         cout<< fixed << outStr<<endl;

         // Open the file in append mode (std::ios::app)
         std::ofstream outFile(solFile,std::ios::app);
         if (!outFile)
            cout<<"Error opening output file: "<<solFile<<std::endl;
         else
         {  outFile<<outStr;
            outFile.close();
         }
      }
   }
   cout << "<ENTER> to exit ..."; getchar();
}
