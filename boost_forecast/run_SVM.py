import sklearn.preprocessing as skprep
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

# Prepare the data for training
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:i + time_steps]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def go_svm(ds, look_back=3, verbose=False):
   Xtrain = np.arange(len(ds)-look_back).reshape(1, -1)
   Ytrain = ds[:-look_back].reshape(1, -1)
   sc_X = skprep.StandardScaler()
   sc_y = skprep.StandardScaler()
   Xtrain = Xtrain.reshape(-1, 1)
   Xtrain1 = sc_X.fit_transform(Xtrain)
   Ytrain1 = sc_y.fit_transform(Ytrain)

   regressor = SVR(kernel='rbf',C=250.0, gamma=20, epsilon=0.2)
   regressor.fit(Xtrain1,Ytrain1.flatten())

   ypred1 = regressor.predict(Xtrain1)
   ypred  = sc_y.inverse_transform([ypred1])[0]

   Xtest  = np.arange(len(ds)-look_back,len(ds)).reshape(1, -1)
   Xtest  = Xtest.reshape(-1, 1)
   Xtest_steps = look_back  # Number of test time steps to forecast

   last_train = ds[-look_back] # Last values from the test set to start forecasting

   # predict test values
   ytest = []
   for _ in range(look_back):
      # Predict next value based on the last sequence
      next_pred = regressor.predict(last_train.reshape(1, -1))
      ytest.append(next_pred[0])
      # Update the last sequence by removing the first element and adding the predicted value
      last_train = next_pred

   # Inverse scaling for future forecast
   future_forecast_inv = sc_y.inverse_transform(np.array(ytest).reshape(-1, 1))

   plt.plot(ds, color='red',label="ds", linewidth=3)
   plt.plot(ypred,label="ypred", color='blue')
   plt.plot(ytest,label="ytest", color='y')
   plt.title("SVR")
   plt.legend()
   plt.show()

   return yfore