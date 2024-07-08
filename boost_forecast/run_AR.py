import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt
import sklearn as sk

def go_AR(ds, look_back=3, verbose=False):
   if(look_back!=3):
      print("ERROR, look_back must be 3 in this applicaition")
      return
   x = np.arange(len(ds))
   y = ds #.values
   x_train, x_test = x[:-look_back], x[-look_back:]
   y_train, y_test = y[:-look_back], y[-look_back:]

   p = 7
   model = AutoReg(y_train, lags=p)
   model_fitted = model.fit()
   start = 0
   end = len(x_train) + 3 -1 # Predicting 3 steps ahead (-1 because end included)
   pred = model_fitted.predict(start=start, end=end) # prediction and forecast

   mse = sk.metrics.mean_absolute_error(y_test, pred[-look_back:])
   print("MSE={}".format(mse))
   ypred = pred[:-look_back]
   yfore = pred[-look_back:]

   if verbose:
      plt.plot(y)
      plt.plot(ypred)
      plt.plot([None for x in ypred]+[x for x in yfore])
      plt.show()
   return yfore
