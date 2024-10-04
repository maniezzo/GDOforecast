#include "detequiv.h"


// Read instance data
void StochMIP::readInstance(string& fileName) 
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

// reads the numserver forecasts for each of the boosted series
int StochMIP::readBoostForecasts(string filePath,int nboost)
{  string line;
   int i,j;
   ifstream infile;

   cout << "Reading " << filePath << endl;
   infile.exceptions(ifstream::failbit | ifstream::badbit);
   infile.open(filePath);
   // Read the file line by line
   boostFcasts.resize(n);
   for(i=0;i<n;i++)
      boostFcasts[i].resize(nboost);

   i=0;
   while (i<m && getline(infile, line))
   {  stringstream ss(line);
      string value;

      // Split the line by commas
      j=0;
      while (getline(ss, value, ',') && j<n) 
      {  boostFcasts[i][j] = round(stof(value)); // Convert to integer and add to row
         j++;
      }
      i++;
   }      
   infile.close();

   return 0;
}