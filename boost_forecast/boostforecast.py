import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import run_randomf as rf
import run_AR as ar
import run_xgboost as xgb
import run_sarimax as sar
import run_MLP as mlp
import run_lstm as lstm
import run_SVM as svm
import run_HW as hw
import sys

# forecasts a single future value look_back time instant ahead using the specified mathod
def forecast_value(ds,dslog,method,look_back = 3, verbose = False):

   if(method=="AR"):
      fcast = ar.go_AR(ds[:-look_back], look_back=look_back, verbose=verbose, gridSearch=True)  # AR semplice
   elif (method == "HW"):
      fcast = hw.go_HW(ds[:-look_back], look_back=look_back, verbose=verbose)  # Holt Winters semplice
   elif (method == "randomf"):
      fcast = rf.go_rf(ds[:-look_back], look_back=look_back, verbose=verbose)  # random forest,
   elif(method=="xgboost"):
      fcast = xgb.go_xgboost(ds[:-look_back], look_back=look_back, verbose= verbose)  # XGboost
   elif (method == "arima"):
      fcast = sar.go_sarima(ds[:-look_back], look_back=look_back, autoArima=True, verbose=verbose)  # ARIMA
   elif (method == "MLP"):
      fcast = mlp.go_MLP(ds[:-look_back], look_back=look_back, lr=0.05, niter=1000, verbose=verbose)  # MLP, pytorch
   elif (method == "lstm"):
      fcast = lstm.go_lstm(ds[:-look_back], look_back=look_back, lr=0.05, niter=1000, verbose=verbose)  # MLP, pytorch
   elif(method=="svm"):
      fcast = svm.go_svm(ds[:-look_back],look_back = look_back, verbose=verbose) # svm

   # forecast undiff
   fcast[0] = fcast[0] + dslog[-1]
   fcast[1] = fcast[1] + fcast[0]
   fcast[2] = fcast[2] + fcast[1]
   fvalue = np.exp(fcast[2])

   if verbose:
      plt.plot(dslog, label="dslog")
      plt.plot(range(len(dslog), len(dslog) + 3), fcast, label="fcast")
      plt.legend()
      plt.title(f"forecast method {method}")
      plt.show()
   return fvalue

def main_fcast(name, df, idcustomer=0, model='AR', fback=0, frep=1, nboost=125, verbose=True):
   dbfilePath='../data/results.sqlite'
   sys.path.append('../boosting')
   import sqlite101 as sql
   sql.querySqlite(dbfilePath, model, fback, frep, nboost) # writes the csv files selecting data from db
   with open(f"res_{model}_{nboost}.csv", "w") as fout:
      fout.write("series,attrib,fcast_50,fcast_avg,fcast_05,fcast_95,true,yar,yhw,ysvm,ylstm,ymlp,yrf,yxgb,yarima\n")

   # foreach boosted series forecast
   #for iboostset in len(df): # for each block of boosted series
   for iboostset in range(idcustomer,idcustomer+52):
      bset = pd.read_csv(f"../data/boost{iboostset}.csv", header=None) # no validation data
      fcast_all = np.zeros(len(bset))
      look_back = 3 # solo con questo va

      for idserie in range(bset.shape[0]):
         ds = np.array(bset.iloc[idserie, 0:])  # one series of bootstrap set, diff log values
         if(model == "AR"):      fcast = ar.go_AR(ds, look_back=look_back, verbose=False, gridSearch=True) # (idserie==0)) # AR, validazione nel metodo
         elif(model == "RF"):    fcast = rf.go_rf(ds, look_back=look_back, verbose=False) # (idserie==0)) # random forest, keeping look-back out for validation
         elif(model == "ARIMA"): fcast = sar.go_sarima(ds, look_back=look_back, autoArima=True, verbose=False) #(idserie==0))  # ARIMA

         trueval = bset.iloc[iboostset,-1] # valore vero
         print(f"idserie,{idserie}, true last {trueval} forecast,{fcast[2]}, error {trueval-fcast[2]}\n")

         # forecast: undiff fcast (yfore)
         seedval = np.log( df.iloc[6,iboostset] ) # cheating a bit, I should have saved this somewhere else, maybe
         dslog = np.zeros(len(ds)-look_back)
         dslog[0] = seedval + ds[0]
         for j in range(1,len(dslog)): dslog[j] = ds[j]+dslog[j-1]
         fcast[0] = fcast[0]+dslog[-1]
         fcast[1] = fcast[1]+fcast[0]
         fcast[2] = fcast[2]+fcast[1]

         fvalue = np.exp(fcast[2])
         fcast_all[idserie] = fvalue
         print(f"forecast value = {fvalue}")

         if idserie == 0:
            if verbose:
               plt.plot(dslog,label="dslog")
               plt.plot(range(len(dslog),len(dslog)+3),fcast,label="fcast")
               plt.legend()
               plt.title(f"series {idserie}")
               plt.show()

            ds = np.exp(dslog)
            if verbose:
               plt.plot(range(7,7+len(ds)),ds,'b:',label="recostruction",linewidth=5)
               plt.plot(df.iloc[:-look_back,iboostset],'r',label="xtrain",linewidth=2)
               plt.plot(range(7+len(ds),7+len(ds)+3),np.exp(fcast),"g",label="fcast",linewidth=2)
               plt.legend()
               plt.title(f"Check series {iboostset}")
               plt.show()
            print(f"Check forecast = {fvalue}")

      # previsione = media
      fcast_all = np.sort(fcast_all)
      fcast_avg = np.average(fcast_all)

      # intervallo = 5 - 95
      fcast_05 = fcast_all[int(len(fcast_all)/100*5)]   # 125/100*5
      fcast_95 = fcast_all[int(len(fcast_all)/100*95)]  #
      fcast_50 = fcast_all[int(len(fcast_all)/100*50)]

      # non-bootssrap point forecasts
      fForeCast = True
      if fForeCast:
         ds = np.array(bset.iloc[0, 1:])  # one series of bootstrap set, diff log values, remove first one
         yar    = forecast_value(ds,dslog,method="AR",look_back=look_back,verbose=verbose)
         yhw    = forecast_value(ds,dslog,method="HW",look_back=look_back,verbose=verbose)
         ysvm   = forecast_value(ds,dslog,method="svm",look_back=look_back,verbose=verbose)
         ylstm  = forecast_value(ds,dslog,method="lstm",look_back=look_back,verbose=verbose)
         ymlp   = forecast_value(ds,dslog,method="MLP",look_back=look_back,verbose=verbose)
         yrf    = forecast_value(ds,dslog,method="randomf",look_back=look_back,verbose=verbose)
         yxgb   = forecast_value(ds,dslog,method="xgboost",look_back=look_back,verbose=verbose)
         yarima = forecast_value(ds,dslog,method="arima",look_back=look_back,verbose=verbose)

      # validazione previsione algoritmi

      # distribution of forecasts, plt histogram:
      if verbose:
         plt.hist(fcast_all, color='lightgreen', ec='black', bins=15)
         plt.xlabel("Values")
         plt.ylabel("Frequency")
         plt.axvline(x=fcast_50,  color='gray', linestyle='-', linewidth=3, label='Median')
         plt.axvline(x=fcast_avg, color='gray', linestyle='--', label='Average')
         plt.axvline(x=fcast_05,  color='gray', linestyle='-', label='0.05')
         plt.axvline(x=fcast_95,  color='gray', linestyle='-', label='0.95')
         plt.axvline(x=df.iloc[-1,iboostset],  color='red',  linestyle='-', label='true val')
         plt.axvline(x=yar,   color='b', linestyle=':', label='AR')
         plt.axvline(x=yhw,   color='g', linestyle=':', label='HW')
         plt.axvline(x=ysvm,  color='c', linestyle=':', label='SVM')
         plt.axvline(x=ylstm, color='m', linestyle=':', label='LSTM')
         plt.axvline(x=ymlp,  color='y', linestyle=':', label='MLP')
         plt.axvline(x=yrf,   color='r', linestyle=':', label='RF')
         plt.axvline(x=yxgb,  color='k', linestyle=':', label='XGB')
         plt.axvline(x=yarima, color='purple', linestyle=':', label='ARIMA')
         plt.title(f"Distribution of forecast Values, series {iboostset}")
         plt.legend()
         plt.show()

      print("series","attrib","true","fcast_50","fcast_avg","fcast_05","fcast_95","yar","yhw","ysvm","ylstm","ymlp","yrf","yxgb","yarima")
      print(iboostset, model, df.iloc[-1,iboostset], fcast_50, fcast_avg, fcast_05, fcast_95, yar, yhw, ysvm, ylstm, ymlp, yrf, yxgb, yarima)
      # Append results to res file
      with open(f"res_{model}_{nboost}.csv", "a") as fout:
         fout.write(f"{iboostset},{model},{df.iloc[-1,iboostset]},{fcast_50},{fcast_avg},{fcast_05},{fcast_95},{yar},{yhw},{ysvm[-1]},{ylstm},{ymlp},{yrf},{yxgb},{yarima}\n")
   print("finito")

if __name__ == "__main__":
   name = "dataframe_nocovid_full"
   df2 = pd.read_csv(f"../{name}.csv", usecols=[i for i in range(1, 53)])
   print(f"Boost forecasting {name}")
   attrib = "rf"  # random resampling, only forcast
   distrib = "AR" # "RF" "ARIMA"
   model="AR"
   fback=0
   frep=1
   nboost=100
   attrib+=distrib
   main_fcast(name, df2.iloc[:-3,:], idcustomer=0, model=model, fback=fback, frep=frep, nboost=175, verbose=False) # actual data only for 45 months
