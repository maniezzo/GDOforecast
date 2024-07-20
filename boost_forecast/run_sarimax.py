import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

def go_sarima(ds, look_back=3, autoArima=False,verbose=False):
   print("PMDARIMA: WATCH OUT! needs numpy 1.26.0 or higher")
   if autoArima:
      model = pm.auto_arima(ds,
                            start_p=0, start_q=0, max_p=2, max_q=2,
                            start_P=0, start_Q=0, max_P=1, max_Q=1,
                            seasonal=True, m=12, d=1, D=None, test='adf',
                            trace=False, error_action='warn', suppress_warnings=True,
                            maxiter = 50, stepwise=True)
      morder     = model.order
      mseasorder = model.seasonal_order
   else:
      #ARIMA(0,1,1)(1,1,0)[12] intercept
      morder = (0,1,1) #model.order
      mseasorder = (1,1,0,12) #model.seasonal_order

   model = pm.arima.ARIMA(morder, seasonal_order=mseasorder,  return_conf_int=True)
   fitted = model.fit(ds)
   print(f"ARIMA: order {morder} mseasorder {mseasorder}")
   #print(model.summary())
   ypred = fitted.predict_in_sample()
   yfore,confint = fitted.predict(n_periods=3,return_conf_int=True)  # forecast
   if verbose:
      plt.plot(ds,label="empyrical")
      plt.plot(ypred,label="model")
      plt.plot([None for x in ypred]+[x for x in yfore],label="forecast")
      plt.title("ARIMA model")
      plt.legend()
      plt.show()
   '''
   # the same, sklearn (dopo autoarima)
   from statsmodels.tsa.statespace.sarimax import SARIMAX
   sarimax_model = SARIMAX(ds, order=morder, seasonal_order=mseasorder, exogenous=externals)
   sfit = sarimax_model.fit()
   sfit.plot_diagnostics(figsize=(10, 6))
   plt.show()
   ypred = sfit.predict(start=0,end=len(ds), exog=externals)
   forewrap=sfit.get_forecast(steps=3,exog=externals[9:12])
   forecast_ci = forewrap.conf_int()
   forecast_val = forewrap.predicted_mean
   plt.plot(ds)
   plt.plot([x for x in ypred[1:]])
   plt.fill_between(np.linspace(len(ds), len(ds) + 3, 3),
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='k', alpha=.25)
   plt.plot(np.linspace(len(ds), len(ds) + 3, 3), forecast_val)
   plt.title("sklearn")
   plt.show()
   '''
   return yfore
