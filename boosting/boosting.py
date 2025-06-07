import numpy as np
import pandas as pd, sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.regression.linear_model import yule_walker
from scipy.stats import boxcox
from scipy.special import inv_boxcox
import random, copy
import sqlite101 as sql
import pmdarima as pm

'''
This code generates the bosted time series and stores them in sqlite.
Forecasting of these series in in boostforecast
'''

# backcast the first 6 data
def backcast(ts,p,verbose=True):
   # Reverse the time series data
   reversed_ts = ts[::-1]
   model = AutoReg(reversed_ts[:-1], lags=p)
   model_fitted = model.fit()

   # Generate backcasted predictions
   start = 0
   end = len(ts) - 1
   backcasted_predictions = model_fitted.predict(start=start, end=end)

   # Reverse the backcasted predictions to the original order
   backcasted_predictions = backcasted_predictions[::-1]

   # Plot the actual data and backcasted predictions
   if verbose:
      plt.figure(figsize=(12, 6))
      plt.plot(ts[1:], label='Actual Data')
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

# alcune prove iniziali, ininfluenti sul risultato
def recolor_check():
   p = 3 # only 3 for check
   ts = np.array([1.4,1,1.9,1.6,1.3,1.4,1.9,1.2])
   model = AutoReg(ts, lags=p, trend='n')
   model_fitted = model.fit()
   phiHin = model_fitted.params
   predictions2 = model_fitted.predict(start=p, end=len(ts)-1)
   check2 = [phiHin[0]*ts[i-1]+phiHin[1]*ts[i-2]+phiHin[2]*ts[i-3] for i in range(3,len(ts))]

   # backward filtering
   tsflip = np.flip(ts)
   model = AutoReg(tsflip, lags=p, trend='n')
   model_fitted = model.fit()
   phiHer = model_fitted.params
   predictions1 = np.flip( model_fitted.predict(start=p, end=len(ts)-1) )
   check1 = [phiHer[0]*tsflip[i-1]+phiHer[1]*tsflip[i-2]+phiHer[2]*tsflip[i-3] for i in range(5,len(ts))]

   pred = np.concatenate((predictions1[:p+1] ,predictions2)) # two (( are mandatory
   residuals = np.array([ts[i]-pred[i] for i in range(len(ts))])
   res_repetition = random.choices(residuals, k=len(residuals))  # extraction with repetition

   Xfor = np.zeros(len(ts))
   for i in range(p,len(ts)):
      Xfor[i] = sum([phiHin[k]*pred[i-k] for k in range(p)]) + res_repetition[i]

   for i in range(p):
      Xfor[i] = sum([phiHer[k]*pred[i+k] for k in range(p)]) + res_repetition[i] # i primi p valori, con i phi her

   # recoloring
   Xnew = np.zeros(len(ts))
   for i in range(len(ts)):
      Xnew[i] = Xfor[i] + res_repetition[i]
   return Xnew

# computes AIC
def aic(y, ar_coeffs, p):
   # Calculate residuals
   ypred = np.zeros(len(y))
   for t in range(p, len(y)):
      ypred[t] = np.dot(ar_coeffs, y[t-p:t][::-1]) # [::-1] reverses
   residuals = y[p:] - ypred[p:]
   # Variance of the residuals
   varRes = np.var(residuals)
   # Log-likelihood
   n = len(y) - p
   log_likelihood = -n / 2 * (np.log(2 * np.pi * varRes) + 1)
   # Number of parameters (AR coefficients + variance) - should be +2 if non zero mean
   k = p + 2
   # AIC calculation
   AIC = -2 * log_likelihood + 2*k
   return AIC

def main_boosting(name,df,backCast = True, repetition=True, nboost=75,p=7,verbose=True,bmodel="AR"):
   recolor_check()
   # plot all series
   if verbose:
      for idserie in range(len(df)):
         plt.plot(df.iloc[:,idserie])
         plt.title(name)
      plt.show()

   tablePreProc(df)
   sql.deleteSqLite("..\\data\\results.sqlite", bmodel, nboost, (1 if repetition else 0))

   for idserie in range(0,52):
      ts = df.iloc[:-3, idserie]

      # log diff della serie
      tslog = np.log(ts)
      tslogdiff = np.zeros(len(ts))
      tslogdiff[0] = tslog[0]
      #for i in range(len(tslogdiff) - 1, 0, -1):
      for i in range(1,len(tslogdiff)):
         tslogdiff[i] = float(tslog[i] - tslog[i-1])
      tslogdiff = np.array(tslogdiff)
      avglogdiff = np.average(tslogdiff[1:])
      stdlogdiff = np.std(tslogdiff[1:])
      adflogdiff = sm.tsa.stattools.adfuller(tslogdiff[1:], maxlag=None, regression='ct', autolag='AIC')[1]
      print(f"iter {idserie}: chack ts[0]={np.exp(tslogdiff[0])} (<->{ts[0]}), ts[1]={np.exp(tslogdiff[1]+tslogdiff[0])} (<->{ts[1]})")
      start = 0  # if no backcasting the first p will be later deleted
      end = len(tslogdiff) - 1  # + 3  # Predicting 3 steps ahead
      y = tslogdiff

      if(bmodel=='AR'):
         model = AutoReg(tslogdiff, lags=p)
         model_fitted = model.fit()
         ypred = model_fitted.predict(start=start, end=end)
      elif(bmodel=="YW"):
         intercept = np.mean(y) # should be 0
         y2 = y - intercept # Center the series in case of non-zero mean
         if abs(intercept) < 0: # remove check significance different p
            for k in range(2,8):
               # Estimate AR coefficients using Yule-Walker equations
               coeff, sigma = yule_walker(y, order=k)
               # sigma is an estimate of the white noise variance in the time series
               phi = -coeff # Yule-Walker returns negative coefficients
               #print(f'phi: {phi}')
               #print(f'sigma: {sigma}')
               AIC = aic(y, phi, k)
               print(f'k: {k} AIC: {AIC}')
         coeff, _ = yule_walker(y2, order=p, method='mle', df=p+2)
         phi = -coeff # Yule-Walker returns negative coefficients
         ypred = np.zeros(len(y))
         for t in range(p, len(y)):
            ypred[t] = np.dot(coeff, y[t-p:t][::-1])  # [::-1] reverses
         ypred[p:] = ypred[p:] + intercept
         #print(f"ypred {ypred}")
      elif(bmodel=='ARIMA'):
         print("ARIMA")
         model = pm.auto_arima(tslogdiff,
                               start_p=0, start_q=0, max_p=2, max_q=2,
                               start_P=0, start_Q=0, max_P=1, max_Q=1,
                               seasonal=False, m=12, d=1, D=None, test='adf',
                               trace=False, error_action='warn', suppress_warnings=True,
                               maxiter=50, stepwise=True)
         morder = model.order
         mseasorder = model.seasonal_order
         model = pm.arima.ARIMA(morder, seasonal_order=mseasorder, return_conf_int=True)
         fitted = model.fit(tslogdiff)
         print(f"ARIMA: order {morder} mseasorder {mseasorder}")
         # print(model.summary())
         ypred = fitted.predict_in_sample()
         yfore, confint = fitted.predict(n_periods=3, return_conf_int=True)  # forecast
         return
      else:
         print(f"Unavailable model {bmodel}, exiting")
         return

      if verbose and idserie==0:
         plt.figure(figsize=(12, 6)) # Plot of actual data and predictions
         plt.plot(range(1,len(y)),y[1:], label='Series Data')
         plt.plot(range(1,len(y)),ypred[1:], label='Predicted Data', color='red')
         plt.axvline(x=p-1, color='gray', linestyle='--', label='Prediction Start')
         plt.title("Predicted vs. actual")
         plt.legend()
         plt.show()

      # backcasting
      if(backCast):
         head = backcast(tslogdiff,p,verbose)
         ypred[0:p] = head[0:p]
         if verbose:
            plt.figure(figsize=(12, 6)) # Plot of actual data and predictions
            plt.plot(tslogdiff[1:], label='Actual Data')
            plt.plot(ypred, label='Predicted Data', color='red')
            plt.axvline(x=start, color='gray', linestyle='--', label='Prediction Start')
            plt.title("combined")
            plt.legend()
            plt.show()
      else:
         start = p

      # residui
      residuals = np.zeros(len(tslogdiff))
      for i in range(0,len(tslogdiff)):
         residuals[i] = tslogdiff[i] - ypred[i]
      if verbose and idserie==0:
         plt.plot(residuals[start:],label="residuals")
         plt.title("residuals")
         plt.legend()
         plt.show()

      # acf and ljung box test
      if verbose and idserie==0:
         plt.rcParams.update({'figure.figsize': (9,7), 'figure.dpi': 120})
         fig, axes = plt.subplots(1, 2, sharex=True)
         axes[0].plot(residuals[start:]);
         axes[0].set_title('Original Series')
         plot_acf(residuals[start:], ax=axes[1])
         plt.show()

      res = sm.stats.diagnostic.acorr_ljungbox(residuals[start:],model_df=p)
      print(f"Ljung box lb_pvalue {res.lb_pvalue[p-1]}")

      # boost, data generation if residuals are random enough
      denoised = np.array([tslogdiff[i] - residuals[i] for i in range(start,len(residuals))]) # aka predictions
      ypred = ypred[start:] # in case of no backcasting
      residuals   = residuals[start:]
      boost_set   = np.zeros(nboost*len(residuals)).reshape(nboost,len(residuals))

      # ------------------------------------------------------------------ generate nboost series
      for iboost in range(nboost):
         if repetition:
            randResiduals = random.choices(residuals, k=len(residuals))  # extraction with repetition
         else:
            randResiduals = np.random.permutation(residuals)             # scramble residuals

         if (iboost==0):   # for checking purposes
            randResiduals = residuals
         for j in range(len(randResiduals)):
            boost_set[iboost,j] = ypred[j] + randResiduals[j]

         # Reconstruction, invert preprocessing
         if iboost==0 and idserie==0:
            # first, reconstruct diff of logs
            lndiff = np.zeros(len(ts))
            for j in range(p): lndiff[j] = tslogdiff[j] # these I must copy
            for j in range(p,len(ts)): lndiff[j] = boost_set[iboost,j-p]
            # second, reconstruct logs
            lnts = np.zeros(len(ts))
            lnts[0] = tslogdiff[0]
            for j in range(1,len(ts)): lnts[j] = lndiff[j] + lnts[j-1]
            rects = np.exp(lnts)
            plt.plot(ts,'r',label='empyrical',linewidth=3)
            plt.plot(rects,':b',label='reconstructed',linewidth=3)
            plt.legend()
            plt.title('check: reconstruction')
            plt.show()

      if verbose and idserie==0:
         for i in range(10):
            plt.plot(boost_set[i,1:])
         plt.title(f"boosted (10), series {idserie}")
         plt.ylim(3*min(boost_set[0,1:]),3*max(boost_set[0,1:]))
         plt.show()

      attrib  = "r" if repetition else "s"  # repetition or scramble
      attrib += "b" if backCast else "f"    # backcast or forecast only (shorter)
      np.savetxt(f"..\\data\\boost{idserie}.csv", boost_set, delimiter=",")
      with open(f"..\\data\\boostset_config.txt", "w") as f:
         f.write(F"model:{bmodel} backcasting:{backCast} repeated extractions: {repetition} num boostr. series:{nboost}")

      # create table boost(id integer primary key autoincrement, model text, nboost int, idseries int, series text)
      fback = 1 if backCast   else 0
      frep  = 1 if repetition else 0
      sql.insertSqlite("..\\data\\results.sqlite", bmodel,fback, frep, nboost, idserie, boost_set)

      # ricostruzione, controllo
      if backCast and verbose and idserie==0:
         #tscheck = np.zeros(len(tslogdiff))
         bocheck0 = np.zeros(len(tslogdiff)) # check residuals
         bocheck1 = np.zeros(len(tslogdiff)) # check rand
         prcheck  = np.zeros(len(tslogdiff))
         #tscheck[0] = tslogdiff[0]
         bocheck0[0] = boost_set[0,0]
         bocheck1[0] = boost_set[1,0]
         prcheck[0]  = tslogdiff[0]
         for i in range(1,boost_set.shape[1]):
            #tscheck[i] = tslogdiff[i] + tscheck[i-1]
            bocheck0[i] = boost_set[0,i] + bocheck0[i - 1]
            bocheck1[i] = boost_set[1,i] + bocheck1[i - 1]
            prcheck[i]  = ypred[i] + prcheck[i - 1]
         #tscheck = np.exp(tscheck)
         bocheck0 = np.exp(bocheck0)
         bocheck1 = np.exp(bocheck1)
         prcheck  = np.exp(prcheck)
         #plt.plot(tscheck,'g:',label="ts check",linewidth=5)
         plt.plot(ts,'r',label="empyrical",linewidth=3)
         plt.plot(bocheck0,'b:',label="boost check residuals",linewidth=5)
         plt.plot(bocheck1,label="boost check rand",linewidth=3)
         plt.plot(prcheck,label="predictions")
         plt.legend()
         plt.title("check")
         plt.show()

   print("finito")
   sys.exit()

if __name__ == "__main__":
   name = "dataframe_nocovid_full"
   df2 = pd.read_csv(f"../data/{name}.csv", usecols = [i for i in range(1,53)])
   print(f"Boosting {name}")
   #sql.createSqlite("..\\data\\results.sqlite")
   p = 5
   main_boosting(name,df2.iloc[:-3,:], backCast=False, repetition=True, nboost = 75, p=p, verbose=True, bmodel="YW") # last 3 were original forecasts
