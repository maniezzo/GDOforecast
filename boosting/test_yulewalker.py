from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.stattools import pacf
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.stattools import adfuller
import statsmodels as sm
import matplotlib.pyplot as plt
import numpy as np
import random as rnd
from statsmodels.tsa.ar_model import AutoReg

test = np.zeros(100)
test[0] = 5
for i in range(1,100):
   test[i] = 0.7*test[i-1] + rnd.random()-0.5
phi, c = yule_walker(test, 1, method='mle')
print(f'phi: {-phi}')
print(f'c: {c}')
valid = np.zeros(199)
valid[0] = c
for i in range(1,199):
   valid[i] = phi[0]*valid[i-1]
plt.plot(test)
plt.plot(valid)
plt.show()


y = np.array([5.5,6.,6.,6.5,5.,5.,6.,6.,6.,9.,8.5,13.,13.5,11.5])
print(y)
#plot_acf(y);
#plot_pacf(y);
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

ypred = [c+phi[0]*y[t-1]+phi[1]*y[t-2] for t in range(2,len(y))]
# plot
plt.plot(y,label="empirical")
plt.plot(range(2,len(y)),predictions[2:], color='red',label="AR(2) fit")
plt.plot( range(2,len(y)),ypred,color="green",label="yule walker")
plt.legend()
plt.show()

print("fine")

