import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import sklearn as sk

def go_HW(ds, look_back=3, verbose=False):
   if(look_back!=3):
      print("ERROR, look_back must be 3 in this applicaition")
      return
   x = np.arange(len(ds))
   y = ds #.values
   x_train, x_test = x[:-look_back], x[-look_back:]
   y_train, y_test = y[:-look_back], y[-look_back:]

   # fit model
   model = ExponentialSmoothing(y_train, seasonal_periods=12, trend="add",
                                seasonal="add",
                                damped_trend=False,
                                use_boxcox=False,
                                initialization_method="estimated")
   hwfit = model.fit()
   # make forecast
   ypred = hwfit.predict(0, len(x_train)-1)
   ytest = hwfit.predict(len(x_train),len(ds)-1)
   yfore = hwfit.predict(len(ds), len(ds)+look_back-1)

   mse = sk.metrics.mean_absolute_error(y_test, ytest)
   print("MSE={}".format(mse))

   if verbose:
      plt.plot(y,label="empyrical")
      plt.plot(ypred,label="train maodel")
      plt.plot([None for x in ypred]+[x for x in ytest],label="test maodel")
      plt.plot(range(len(ds),len(ds)+look_back),yfore,label="forecast")
      plt.title("Forecast method Holt Winters")
      plt.legend()
      plt.show()
   return yfore
