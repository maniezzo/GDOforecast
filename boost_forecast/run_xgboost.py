import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def go_xgboost(ds,look_back=3,verbose=False):
   if(look_back!=3):
      print("ERROR, look_back must be 3 in this application")
      return
   x = np.arange(len(ds))
   y = ds #.values
   x_train, x_test = x[:-look_back], x[-look_back:]
   y_train, y_test = y[:-look_back], y[-look_back:]

   x_train = np.vstack(x_train)
   y_train = np.vstack(y_train)
   x_test = np.vstack(x_test)
   y_test = np.vstack(y_test)
   # fit model
   model = XGBRegressor(objective='reg:squarederror', n_estimators=1500)

   model.fit(x_train, y_train)
   ypred = model.predict(x_train)
   yfore = np.zeros(look_back)
   # rolling forecast, valid only if look_back=3
   for i in np.arange(look_back):
      yfore[i] = model.predict(x_test)[0]
      x_test[0,0] = x_test[1,0]
      x_test[1,0] = x_test[2,0]
      x_test[2,0] = yfore[i]

   if verbose:
      plt.plot(y,label="empyrical")
      plt.plot(ypred,label="model")
      plt.plot([None for x in ypred]+[x for x in yfore],label="forecast")
      plt.title("Random forest model")
      plt.legend()
      plt.show()
   return yfore
