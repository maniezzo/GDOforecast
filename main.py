import os
import numpy as np, pandas as pd
import run_sarimax as s
import run_MLP as mlp
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

   goSarimax = False
   if goSarimax:
      for idserie in np.arange(5): #numseries):
         fsarimax = s.sarima(df[df.columns[idserie]],indices,autoArima=True)
         print(f"idserie={idserie} - forecast {fsarimax[2]}")

   goMLP = True
   if goMLP:
      idserie = 0
      for idserie in np.arange(5): #numseries):
         fmlp =  mlp.go_MLP(df[df.columns[idserie]],indices,look_back=3)
         print(f"idserie={idserie} - forecast {fmlp[2]}")

   pass
