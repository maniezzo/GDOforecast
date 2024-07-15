import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def go_xgboost(ds,indices,look_back=3):
   if(look_back!=3):
      print("ERROR, look_back must be 3 in this application")
      return
   x = np.arange(len(ds))
   y = ds.values
   x_train, xtest = x[:-look_back], x[-look_back:]
   y_train, ytest = y[:-look_back], y[-look_back:]

   x_train = np.vstack(x_train)
   y_train = np.vstack(y_train)
   xtest = np.vstack(xtest)
   ytest = np.vstack(ytest)
   # fit model
   model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)

   model.fit(x_train, y_train)
   ypred = model.predict(x_train)
   yfore = np.zeros(look_back)
   # rollong forecast
   for i in np.arange(look_back):
      yfore[i] = model.predict(xtest)[0]
      xtest = np.append(xtest[1:],yfore[i])

   plt.plot(ds.values)
   plt.plot(ypred)
   plt.plot([None for x in ypred]+[x for x in yfore])
   plt.show()
   return yfore