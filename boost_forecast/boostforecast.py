import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_randomf as rf
import run_AR as ar

def main_fcast(name, df):
   # foreach boosted series forecast
   #for iboostset in len(df): # for each block of boosted series
   idserie = 10
   for iboostset in range(idserie,idserie+1):
      bset = pd.read_csv(f"../data/boost{iboostset}.csv",header=None)
      fcast_all = np.zeros(len(bset))
      look_back = 3 # solo con questo va

      for idserie in range(len(bset)):
         ds = np.array(bset.iloc[idserie, :])  # one series of bootstrap set
         dlog = np.log(ds)
         fcast = rf.go_rf(dlog[:-look_back],look_back=look_back, verbose= (idserie==0))  # random forest
         #fcast = ar.go_AR(dlog,look_back=look_back, verbose= (idserie==0)) # AR, validazione nel metodo
         trueval = dlog[-1] # valore vero
         print(f"idserie,{idserie},forecast,{fcast[2]}, error {trueval-fcast[2]}\n")

         fvalue = np.exp(fcast[2])
         fcast_all[idserie] = fvalue
         print(f"forecast value = {fvalue}")

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
   main_fcast(name, df2) # momentarily useless arguments
