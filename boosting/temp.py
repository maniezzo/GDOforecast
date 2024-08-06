from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.stattools import adfuller
import statsmodels as sm
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

y = np.array([5.5,6.,6.,6.5,5.,5.,6.,6.,6.,9.,8.5,13.,13.5,11.5])
print(y)
plot_acf(y);
plot_pacf(y);
pacf_coef_AR2 = pacf(y)
print(pacf_coef_AR2)
phi, c = yule_walker(y, 2, method='mle')
print(f'phi: {-phi}')
print(f'c: {c}')

model = AutoReg(y, lags=2)
model_fit = model.fit()
coef = model_fit.params
print(f"AR params {coef}")
# walk forward over time steps in test
window = 2
history = y[len(y)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(y)):
  length = len(history)
  lag = [history[i] for i in range(length-window,length)]
  yhat = coef[0]
  for d in range(window):
    yhat += coef[d+1] * lag[window-d-1]
  obs = y[t]
  predictions.append(yhat)
  history.append(obs)
  print('predicted=%f, expected=%f' % (yhat, obs))
# plot
plt.plot(y)
plt.plot(predictions, color='red')
plt.show()

ypred = [c+phi[0]*y[t-1]+phi[1]*y[t-2] for t in range(2,len(y)+1)]

