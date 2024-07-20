import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import sklearn as sk

def ARlog_likelyhood(y,coeff,p):
   n = len(y)
   residuals = []
   for t in range(p, n): # cannot use the first p
      #ytpred = coeff[0]
      #for i in range(p):
      #   ytpred += coeff[i+1]*y[t-p+i]
      ytpred = coeff[0] + np.sum(coeff[1:]*y[t-p:t])
      residuals.append(y[t] - ytpred)
   sigma = np.std(residuals) # standard deviation of the residuals

   residuals = np.array(residuals)
   n_residuals = len(residuals)

   add1 = -n_residuals / 2 * np.log(2 * np.pi * sigma ** 2) # nres/2 * 2pi s^2
   add2 = -1 / (2 * sigma ** 2) * np.sum(residuals ** 2)    # -1/(2 s^2) * sum(res^2)

   log_likelihood = add1 + add2
   return log_likelihood

def go_AR(ds, look_back=3, verbose=False, gridSearch = False):
   if(look_back!=3):
      print("ERROR, look_back must be 3 in this applicaition")
      return
   x = np.arange(len(ds))
   y = ds #.values
   x_train, x_test = x[:-look_back], x[-look_back:]
   y_train, y_test = y[:-look_back], y[-look_back:]

   if gridSearch:
      bestp = -1
      best_score = float("inf")
      for p in range(0,5):
         model = AutoReg(y_train, lags=p)
         model_fit = model.fit()

         # Extract the log-likelihood and number of parameters
         log_likelihood = model_fit.llf
         ll2 = ARlog_likelyhood(y_train, model_fit.params, p)
         num_params = model_fit.df_model + 1  # Number of parameters

         # Compute AIC
         aic = 2 * num_params - 2 * log_likelihood
         if aic<best_score:
            best_score = aic
            bestp = p
         print(f'p:{p} AIC: {model_fit.aic} check:{aic}')
      print("Best p:", bestp)
      print("Best score:", best_score)
      p = bestp
   else:
      p = 7

   model = AutoReg(y_train, lags=p)
   model_fitted = model.fit()
   start = 0
   end = len(x_train) + 3 -1 # Predicting 3 steps ahead (-1 because end included)
   pred = model_fitted.predict(start=start, end=end) # prediction and forecast

   mse = sk.metrics.mean_absolute_error(y_test, pred[-look_back:])
   print(f"AR: p={p} MSE={mse}")
   ypred = pred[:-look_back]
   yfore = pred[-look_back:]

   if verbose:
      plt.plot(y)
      plt.plot(ypred)
      plt.plot([None for x in ypred]+[x for x in yfore])
      plt.show()
   return yfore
