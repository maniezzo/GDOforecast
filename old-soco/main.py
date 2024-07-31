import os, json
import numpy as np, pandas as pd
from boosting import boosting as b

def writeseries(df):
   numrows = 48
   cols = df.loc[df["idserie"]==0]["val"].iloc[0:numrows].to_numpy().reshape(numrows,1)
   names = ["cust0"]
   for i in np.arange(1,52,1):
      c = df.loc[df["idserie"]==i]["val"].iloc[0:numrows].to_numpy().reshape(numrows,1)
      cols = np.append(cols,c, axis=1)
      names = np.append(names,f"cust{i}")
   df = pd.DataFrame(cols)
   df.columns = names
   df.to_csv("dataframe_nocovid_full.csv")

if __name__ == "__main__":
   os.chdir(os.path.dirname(os.path.abspath(__file__)))

   # read configuration
   fconf = open('config.json')
   conf = json.load(fconf)
   numSeries = conf["jNumSeries"]
   fgoSarima = conf['jgoSarima']
   fgoMLP   = conf['jgoMLP']
   lrMLP    = conf["jMLPlr"]
   niterMLP = conf["jMLPniter"]
   fgoLSTM   = conf['jgoLSTM']
   lrLSTM    = conf["jLSTMlr"]
   niterLSTM = conf["jLSTMniter"]
   fgoSVM    = conf['jgoSVM']
   fgoXGboost= conf['jgoXGboost']
   fgoRF     = conf['jgoRF']
   fBoost    = conf['jBoost']
   fconf.close()

   #df = pd.read_csv("../serie_nocovid_new.csv")
   #writeseries(df)
   indices = pd.read_csv("indices.csv",header=None).values[0,:]

   df = pd.read_csv("dataframe_nocovid.csv", usecols = [i for i in range(1,53)])
   (numt,numseries) = df.shape

   df2 = pd.read_csv("dataframe_nocovid_full.csv", usecols = [i for i in range(1,53)])
   ref = df2.iloc[len(df2)-1,:].values

   name = "dataframe_nocovid_full"
   print(f"Boosting {name}")
   if(fBoost): b.main_boosting(name,df2)
   del df2

   import run_sarimax as s
   import run_MLP as mlp
   import run_lstm as lstm
   import run_SVM as svm
   import run_xgboost as xgb
   import run_randomf as rf

   idserie = 0
   #from statsmodels.tsa.seasonal import seasonal_decompose
   #result = seasonal_decompose(df["cust0"], model='multiplicative', period=12)
   #result.plot()
   #plt.show()
   fout = open('results.csv', 'w')

   if fgoSarima:
      for idserie in np.arange(numseries):
         fsarimax = s.sarima(df[df.columns[idserie]], indices, autoArima=True)
         if isinstance(fsarimax, pd.Series):
            fsarimax = fsarimax.values
         print(f"idserie={idserie} - forecast {fsarimax[2]}")
         fout.write(f"sarimax,idserie,{idserie},forecast,{fsarimax[2]}, error {ref[idserie]-fsarimax[2]}\n")

   if fgoMLP:
      idserie = 0
      for idserie in np.arange(numSeries):
         fmlp =  mlp.go_MLP(df[df.columns[idserie]],indices,look_back=3,lr=lrMLP,niter=niterMLP)
         print(f"idserie={idserie} - forecast {fmlp[2]}")
         fout.write(f"mlp,idserie,{idserie},forecast,{fmlp[2]}, error {ref[idserie]-fmlp[2]}\n")

   if fgoLSTM:
      idserie = 0
      for idserie in np.arange(numSeries):
         flstm =  lstm.go_lstm(df[df.columns[idserie]],indices,look_back=3,lr=lrLSTM,niter=niterLSTM)
         print(f"idserie={idserie} - forecast {flstm[2]}")
         fout.write(f"lstm,idserie,{idserie},forecast,{flstm[2]}, error {ref[idserie]-flstm[2]}\n")

   if fgoSVM:
      idserie = 0
      for idserie in np.arange(numSeries):
         fsvm =  svm.go_svm(df[df.columns[idserie]],indices)
         print(f"idserie={idserie} - forecast {fsvm[2]}")
         fout.write(f"svm,idserie,{idserie},forecast,{fsvm[2]}, error {ref[idserie]-fsvm[2]}\n")

   if fgoXGboost:
      idserie = 0
      for idserie in np.arange(numSeries):
         fxgboost = xgb.go_xgboost(df[df.columns[idserie]],indices)
         print(f"idserie={idserie} - forecast {fxgboost[2]}")
         fout.write(f"xgboost,idserie,{idserie},forecast,{fxgboost[2]}, error {ref[idserie]-fxgboost[2]}\n")

   if fgoRF:
      idserie = 0
      for idserie in np.arange(numSeries):
         frf = rf.go_rf(df[df.columns[idserie]],indices,look_back=3)
         print(f"idserie={idserie} - forecast {frf[2]}")
         fout.write(f"rndfrst,idserie,{idserie},forecast,{frf[2]}, error {ref[idserie]-frf[2]}\n")

   fout.close()
   pass
