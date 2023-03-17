import os, json
import numpy as np, pandas as pd
import run_sarimax as s
import run_MLP as mlp
import run_lstm as lstm

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
   fconf.close()

   #df = pd.read_csv("../serie_nocovid_new.csv")
   #writeseries(df)
   indices = pd.read_csv("indices.csv",header=None).values[0,:]

   df = pd.read_csv("dataframe_nocovid.csv", usecols = [i for i in range(1,53)])
   (numt,numseries) = df.shape

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
         fout.write(f"sarimax,idserie,{idserie},forecast,{fsarimax[2]}\n")

   if fgoMLP:
      idserie = 0
      for idserie in np.arange(numSeries):
         fmlp =  mlp.go_MLP(df[df.columns[idserie]],indices,look_back=3,lr=lrMLP,niter=niterMLP)
         print(f"idserie={idserie} - forecast {fmlp[2]}")
         fout.write(f"mlp,idserie,{idserie},forecast,{fmlp[2]}\n")

   if fgoLSTM:
      idserie = 0
      for idserie in np.arange(numSeries):
         flstm =  lstm.go_lstm(df[df.columns[idserie]],indices,look_back=3,lr=lrLSTM,niter=niterLSTM)
         print(f"idserie={idserie} - forecast {flstm[2]}")
         fout.write(f"lstm,idserie,{idserie},forecast,{flstm[2]}\n")

   fout.close()
   pass
