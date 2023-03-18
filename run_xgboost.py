import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

def go_xgboost(ds):
   x = ds.iloc[:, 1:]
   y = ds.iloc[:, 0]
   x_train, xtest = x[:-12], x[-12:]
   y_train, ytest = y[:-12], y[-12:]

   # fit model
   model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
   model.fit(x_train, y_train)
   # make a one-step prediction
   yhat = model.predict(xtest)
   yfore = yhat
   return yfore