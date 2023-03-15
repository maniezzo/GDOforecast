import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

def sarima(ds,indices,autoArima=False):
   externals = np.array([indices[i % 12] for i in np.arange(len(ds))]).reshape(-1, 1)
   if autoArima:
      model = pm.auto_arima(ds.values, exogenous=externals, start_p=1, start_q=1,
                            test='adf', max_p=3, max_q=3, m=12,
                            start_P=0, seasonal=True,
                            d=None, D=1, trace=True,
                            error_action='ignore',
                            suppress_warnings=True,
                            stepwise=True)  # False full grid
   #ARIMA(0,1,1)(2,1,0)[12] intercept
   morder = (0,1,1) #model.order
   mseasorder = (2,1,0,13) #model.seasonal_order

   model = pm.arima.ARIMA(morder, seasonal_order=mseasorder, exogenous=externals, return_conf_int=True)
   fitted = model.fit(ds)
   print(model.summary())
   yfore = fitted.predict(n_periods=3,exog=externals[9:12])  # forecast
   ypred = fitted.predict_in_sample(exog=externals)
   plt.plot(ds.values)
   plt.plot(ypred)
   plt.plot([None for i in ypred] + [x for x in yfore])
   plt.show()

   from statsmodels.tsa.statespace.sarimax import SARIMAX
   sarimax_model = SARIMAX(ds, order=morder, seasonal_order=mseasorder, exogenous=externals)
   sfit = sarimax_model.fit()
   sfit.plot_diagnostics(figsize=(10, 6))
   plt.show()
   ypred = sfit.predict(start=0,end=len(ds), exog=externals)
   plt.plot(ds)
   plt.plot(ypred)
   plt.show()
   forewrap=sfit.get_forecast(steps=3,exog=externals[9:12])
   forecast_ci = forewrap.conf_int()
   forecast_val = forewrap.predicted_mean
   plt.plot(ds)
   #plt.fill_between(np.linspace(len(ds), len(ds) + 3, 3),
   #                 forecast_ci[:, 0],
   #                 forecast_ci[:, 1], color='k', alpha=.25)
   plt.plot(np.linspace(len(ds), len(ds) + 3, 3), forecast_val)
   plt.show()
