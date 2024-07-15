import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_randomf as rf
import run_AR as ar

def main_fcast(name, df):
   idserie = 19 # this to test only one series
   # foreach boosted series forecast
   #for iboostset in len(df): # for each block of boosted series
   for iboostset in range(idserie,idserie+1):
      bset = pd.read_csv(f"../data/boost{iboostset}.csv",header=None) # 42 values, no validation data
      fcast_all = np.zeros(len(bset))
      look_back = 3 # solo con questo va

      for idserie in range(len(bset)):
         ds = np.array(bset.iloc[idserie, 1:])  # one series of bootstrap set, diff log values, remove first one
         #fcast = rf.go_rf(ds[:-look_back],look_back=look_back, verbose= (idserie==0))  # random forest, keeping look-back out for validation
         fcast = ar.go_AR(ds[:-look_back],look_back=look_back, verbose= (idserie==0)) # AR, validazione nel metodo
         trueval = bset.iloc[idserie,-1] # valore vero
         print(f"idserie,{idserie}, true last {trueval} forecast,{fcast[2]}, error {trueval-fcast[2]}\n")

         # forecast undiff
         dslog = np.zeros(len(ds)+1)
         dslog[0] = bset.iloc[idserie,0]
         for j in range(len(ds)): dslog[j+1] = ds[j]+dslog[j]
         fcast[0] = fcast[0]+dslog[-1]
         fcast[1] = fcast[1]+fcast[0]
         fcast[2] = fcast[2]+fcast[1]

         fvalue = np.exp(fcast[2])
         fcast_all[idserie] = fvalue
         print(f"forecast value = {fvalue}")

         if idserie == 0:
            plt.plot(dslog,label="dslog")
            plt.plot(range(len(dslog),len(dslog)+3),fcast,label="fcast")
            plt.legend()
            plt.title(f"series {idserie}")
            plt.show()

            ds = np.exp(dslog)
            plt.plot(ds,'b:',label="recostruction",linewidth=5)
            plt.plot(df.iloc[:,iboostset],'r',label="xtrain",linewidth=2)
            plt.plot(range(len(ds),len(ds)+3),np.exp(fcast),"g",label="fcast",linewidth=2)
            plt.legend()
            plt.title(f"Check series {iboostset}")
            plt.show()
            print(f"Check forecast = {fvalue}")

      # previsione = media
      fcast_all = np.sort(fcast_all)
      fcast_avg = np.average(fcast_all)

      # intervallo = 5 - 95
      fcast_05 = fcast_all[5]
      fcast_95 = fcast_all[95]

      # validazione previsione algoritmi

      # distribution of forecasts, plt histogram:
      plt.hist(fcast_all, color='lightgreen', ec='black', bins=15)
      plt.xlabel("Values")
      plt.ylabel("Frequency")
      plt.axvline(x=fcast_avg, color='gray', linestyle='--', label='Average')
      plt.axvline(x=fcast_05,  color='gray', linestyle='--', label='0.05')
      plt.axvline(x=fcast_95,  color='gray', linestyle='--', label='0.95')
      plt.axvline(x=df.iloc[-1,iboostset],  color='red',  linestyle='-', label='true val')
      plt.title(f"Distribution of forecast Values, series {iboostset}")
      plt.legend()
      plt.show()
   print("finito")

if __name__ == "__main__":
   name = "dataframe_nocovid_full"
   df2 = pd.read_csv(f"..\{name}.csv", usecols=[i for i in range(1, 53)])
   print(f"Boost forecasting {name}")
   main_fcast(name, df2.iloc[:-3,:]) # actual data only for 45 months
