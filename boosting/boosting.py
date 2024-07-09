import numpy as np
import pandas as pd, sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import random

# backcast the first 6 data
def backcast(ts,p):
   # Reverse the time series data
   reversed_ts = ts[::-1]
   model = AutoReg(reversed_ts, lags=p)
   model_fitted = model.fit()

   # Generate backcasted predictions
   start = 0
   end = len(ts) - 1
   backcasted_predictions = model_fitted.predict(start=start, end=end)

   # Reverse the backcasted predictions to the original order
   backcasted_predictions = backcasted_predictions[::-1]

   # Plot the actual data and backcasted predictions
   plt.figure(figsize=(12, 6))
   plt.plot(ts, label='Actual Data')
   plt.plot(range(len(ts)), backcasted_predictions, label='Backcasted Data', color='red')
   plt.legend()
   plt.title("backcasted")
   plt.show()
   return backcasted_predictions[0:p]

# calcola avg, std a adf di vari preprocessing di df
def tablePreProc(df):
   lstVals = []
   for idserie in range(df.shape[1]):
      ts = df.iloc[:-3,idserie]
      # tutte le serie, media e varianza originali (m,s2,adf), differenziate, differenziate log, differenziate boxcox (m,s2,adf,lambda)
      # avg, std, adf serie originale
      avgorig = np.average(ts.values)
      stdorig = np.std(ts.values)
      adforig = sm.tsa.stattools.adfuller(ts.values, maxlag=None, regression='ct', autolag='AIC')[1]

      # avg, std, adf serie differenziata
      tsdiff1 = [float(ts[i]-ts[i-1]) for i in range(1,len(ts))]
      tsdiff1.insert(0,ts[0])
      tsdiff1 = np.array(tsdiff1)
      avgdiff1 = np.average(tsdiff1[1:])
      stddiff1 = np.std(tsdiff1[1:])
      adfdiff1 = sm.tsa.stattools.adfuller(tsdiff1[1:], maxlag=None, regression='ct', autolag='AIC')[1]

      # avg, std, adf serie logdiff
      tslogdiff = np.log(ts)
      for i in range(len(tslogdiff)-1,0,-1):
         tslogdiff[i] = float(tslogdiff[i]-tslogdiff[i-1])
      tslogdiff = np.array(tslogdiff)
      avglogdiff = np.average(tslogdiff[1:])
      stdlogdiff = np.std(tslogdiff[1:])
      adflogdiff = sm.tsa.stattools.adfuller(tslogdiff[1:], maxlag=None, regression='ct', autolag='AIC')[1]
      print(f"chack {np.exp(tslogdiff[0])}")

      # avg, std, adf serie box cox diff
      tsBCdiff, BClambda = boxcox(ts)
      for i in range(len(tsBCdiff)-1,0,-1):
         tsBCdiff[i] = tsBCdiff[i] - tsBCdiff[i-1]
      print(f"Box-cox lambda value: {BClambda}")
      avgBCdiff = np.average(tsBCdiff[1:])
      stdBCdiff = np.std(tsBCdiff[1:])
      try:
         adfBCdiff = sm.tsa.stattools.adfuller(tsBCdiff[1:], maxlag=None, regression='ct', autolag='AIC')[1]
      except:
         adfbcdiff = np.nan
      lstVals.append([idserie,avgorig,stdorig,adforig,avgdiff1,stddiff1,adfdiff1,avglogdiff,stdlogdiff,adflogdiff,avgBCdiff,stdBCdiff,adfBCdiff,BClambda])

   dfTable = pd.DataFrame(lstVals,
                          columns=['idserie','avgorig','stdorig','adforig','avgdiff1','stddiff1','adfdiff1','avglogdiff','stdlogdiff','adflogdiff','avgBCdiff','stdBCdiff','adfBCdiff','BClambda'])
   dfTable.to_csv('tab_preproc.csv')
   return

def main_boosting(name,df):
   # plot all series
   for idserie in range(len(df)):
      plt.plot(df.iloc[:,idserie])
      plt.title(name)
   plt.show()

   tablePreProc(df)

   p = 7
   idserie = 0
   # fit AR(p)
   ts = df.iloc[:-3,idserie]

   # serie differenziate di ordine 1 per renderle stazionarie, adf su tutte
   adf = sm.tsa.stattools.adfuller(tsBC, maxlag=None, regression='ct', autolag='AIC', store=False, regresults=False)
   print(f"Dickey-fuller = {adf}")

   tsBC = np.log(ts)

   # Box-Cox of data. Requires positive data, add 1
   #ts1 = ts + 1  # Shifting data to be all positive
   #tsBC, BClambda = boxcox(ts)
   #print(f"Box-cox lambda value: {BClambda}")
   # se lambda piccolo diventa logaritmo (lambda = 0), si lascia il log

   #p 3 -> 7, min in cui box cox viene superato, residui rumore bianco, non correlati
   tsBC = pd.Series(tsBC)
   model = AutoReg(tsBC, lags=p)
   model_fitted = model.fit()

   start = 0
   end = len(tsBC)# + 3  # Predicting 3 steps ahead
   predictions = model_fitted.predict(start=start, end=end)

   plt.figure(figsize=(12, 6)) # Plot of actual data and predictions
   plt.plot(tsBC, label='Box Cox Data')
   plt.plot(predictions, label='Predicted Data', color='red')
   plt.axvline(x=start, color='gray', linestyle='--', label='Prediction Start')
   plt.title("forecasted")
   plt.legend()
   plt.show()

   # no backcasting
   head = backcast(tsBC,p)
   predictions[0:p] = head[0:p]
   plt.figure(figsize=(12, 6)) # Plot of actual data and predictions
   plt.plot(tsBC, label='Actual Data')
   plt.plot(predictions, label='Predicted Data', color='red')
   plt.axvline(x=start, color='gray', linestyle='--', label='Prediction Start')
   plt.title("combined")
   plt.legend()
   plt.show()

   # varianza serie
   residuals = np.zeros(len(tsBC))
   for i in range(len(tsBC)):
      residuals[i] = tsBC[i] - predictions[i]
   plt.plot(residuals,label="residuals")
   #plt.plot(tsBC.values,label="series")
   plt.title("residuals")
   plt.legend()
   plt.show()

   # acf and ljung box test
   plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})
   fig, axes = plt.subplots(1, 2, sharex=True)
   axes[0].plot(residuals);
   axes[0].set_title('Original Series')
   plot_acf(residuals, ax=axes[1])
   plt.show()

   res = sm.stats.diagnostic.acorr_ljungbox(residuals,model_df=p)
   print(f"lb_pvalue {res.lb_pvalue[p+1]}")

   # loop if residuals are random enough
   denoised = np.array([tsBC[i] - residuals[i] for i in range(len(residuals))])
   nboost = 100
   boost_set = np.zeros(nboost*len(residuals)).reshape(nboost,len(residuals))
   # generate nboost series
   for iboost in range(nboost):
      res_scramble   = np.random.permutation(residuals)            # scramble residuals
      res_repetition = random.choices(residuals, k=len(residuals)) # extraction with repetition
      for j in range(len(res_repetition)):
         boost_set[iboost,j] = predictions[j] + res_repetition[j]
   boost_set[0] = tsBC  # first seies is the original one

   # reconstruction, inverse log
   for i in range(nboost):
      for j in range(len(residuals)):
         boost_set[i,j] = np.exp(boost_set[i,j])

   for i in range(10):
      plt.plot(boost_set[i,:])
   plt.title("boosted (10)")
   plt.ylim(10,25)
   plt.show()
   np.savetxt(f"..\\data\\boost{idserie}.csv", boost_set, delimiter=",")


   # Invert the Box-Cox transformation
   #inverted_data = inv_boxcox(tsBC, BClambda)
   newts = np.exp(tsBC)

   print("finito")
   sys.exit()

if __name__ == "__main__":
   name = "dataframe_nocovid_full"
   df2 = pd.read_csv(f"..\{name}.csv", usecols = [i for i in range(1,53)])
   print(f"Boosting {name}")
   main_boosting(name,df2.iloc[:-3,:])