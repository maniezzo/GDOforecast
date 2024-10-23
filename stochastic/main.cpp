#include "deterministic.h"
#include "detequiv.h"
#include "json.h"

string solFile;

// returns lower and upper bounds
int CPXPUBLIC myCallbackFunction(CPXCENVptr env, void* cbdata, int wherefrom, void* cbhandle)
{  int status=0,numsec;
   double bestObjVal=-1, incumbObjVal=-1;
   // Cast the callback handle to our custom data structure
   CallbackData *data = static_cast<CallbackData*>(cbhandle);

   // Get the current time
   auto currentTime = chrono::steady_clock::now();

   // Calculate time elapsed since the last print
   chrono::duration<double> elapsedTime = currentTime - data->lastPrintTime;

   // Check if numsec seconds have passed
   numsec = 600;
   if (elapsedTime.count() >= numsec) 
   {  status = CPXgetcallbackinfo (env, cbdata, wherefrom, CPX_CALLBACK_INFO_BEST_INTEGER, &bestObjVal);
      if ( status ) 
      {  cout<<"error " << status << " in CPXgetcallbackinfo"<<endl;
         status = 1;
         goto TERMINATE;
      }

      status = CPXgetcallbackinfo (env, cbdata, wherefrom, CPX_CALLBACK_INFO_BEST_REMAINING, &incumbObjVal);
      if ( status ) 
      {  cout<<"error " << status << " in CPXgetcallbackinfo"<<endl;
         status = 1;
         goto TERMINATE;
      }

      // Print the bounds (if both were retrieved)
      cout << "-----> " << " elapsed " << elapsedTime.count() << " secs: zub: " << bestObjVal << ", zlb: " << incumbObjVal << endl;

      // Reset the timer
      data->lastPrintTime = currentTime;
      // Open the file in append mode (std::ios::app)
      ofstream outFile(solFile,std::ios::app);
      if (!outFile)
         cout<<"Error opening output file: "<<solFile<<endl;
      else
      {  outFile << "elapsed " << elapsedTime.count() << " zlb " << incumbObjVal << " zub " << bestObjVal << endl;
         outFile.close();
      }
   }
TERMINATE:
   return status; // Zero return indicates success
}

void printResults(string strInst, int numScen, int nboost, int npaid,
                  tuple<int,int,int,float,float,double,double> res)
{
   ostringstream osString;
   osString<<"Instance "   << strInst;
   osString<<" num.scen. " << numScen;
   osString<<" num.boost " << nboost;
   osString<<" num.paid "  << npaid;
   osString<<" status "    << get<0>(res);
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

int main()
{  StochMIP Stoch;
   SingleMIP MIP;
   string instanceFile,distribFile;
   string line,inst;
   stringstream ss;
   int i,status,irep,nrep,nmult;
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
   bool isQtest   = JSV["isQtest"];      // generate q distributions
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
   if(isDetmnst && !isQtest)
   {  
      for(irep=0;irep<nrep;irep++)
      {  ss.str("");
         ss<<irep; // multiple instances only in case all q are 0 (name with no q)
         inst = instanceFile+"_"+ss.str()+".json";
         MIP.readInstance(inst);
         tuple<int,int,int,float,float,double,double> res = MIP.solveMIP(TimeLimit,isVerbose);

         size_t slashPos = inst.rfind('/');
         dotPos = inst.rfind('.');
         string strInst = inst.substr(slashPos+1,dotPos-slashPos-1);
         printResults(strInst,numScen,nboost,0,res);
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
   else if (isQtest)
   {  vector<int> qCostUp,qCostDown;
      tuple<int,int,int,float,float,double,double> res;
      inst = instanceFile+".json";
      size_t slashPos = inst.rfind('/');
      dotPos = inst.rfind('.');
      string strInst = inst.substr(slashPos+1,dotPos-slashPos-1);

      for(irep=0;irep<nrep;irep++)
      {  ss.str("");
         ss<<irep; // multiple instances only in case all q are 0 (name with no q)
         inst = instanceFile+"_"+ss.str()+".json";

         MIP.readInstance(inst);
         tuple<vector<int>,vector<int>> tq = MIP.generateQcosts();
         qCostUp = get<0>(tq);
         qCostDown = get<1>(tq);

         // ---------------------------- increasing costs
         for(i=0;i<MIP.m;i++)
            MIP.qcost[i] = 0;

         // vector of indices
         vector<int> indices(MIP.qcost.size());
         for (i = 0; i < qCostUp.size(); ++i) indices[i] = i;

         // Sort indices based on the values in qCostUp (qcost) by ASC
         sort(indices.begin(), indices.end(), [&qCostUp](int i1, int i2) {
            return qCostUp[i1] < qCostUp[i2];  // Compare values in qCostUp
            });

         res = MIP.solveMIP(TimeLimit,isVerbose);
         printResults(strInst,numScen,nboost,0,res);

         for (i=0;i<MIP.m;i++)
         {  MIP.qcost[indices[i]] = qCostUp[indices[i]];
            res = MIP.solveMIP(TimeLimit, isVerbose);
            printResults(strInst, numScen, nboost,i+1, res);
         }

         // ---------------------------- decreasing costs
         for(i=0;i<MIP.m;i++)
            MIP.qcost[i] = 0;

         // vector of indices
         vector<int> indices(MIP.qcost.size());
         for (i = 0; i < qCostDown.size(); ++i) indices[i] = i;

         // Sort indices based on the values in qCostDown (qcost) by ASC
         sort(indices.begin(), indices.end(), [&qCostDown](int i1, int i2) {
            return qCostDown[i1] < qCostDown[i2];  // Compare values in qCostDown
            });

         for(i=0;i<MIP.m;i++)
            MIP.qcost[i] = qCostDown[i];
         res = MIP.solveMIP(TimeLimit,isVerbose);
         printResults(strInst,numScen,nboost,i+1,res);

         for (i=0;i<MIP.m;i++)
         {  MIP.qcost[indices[i]] = qCostDown[indices[i]];
            res = MIP.solveMIP(TimeLimit, isVerbose);
            printResults(strInst, numScen, nboost,i+1, res);
         }
      }
   }
   else if(!isDetmnst)
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
