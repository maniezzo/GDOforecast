import sklearn.preprocessing as skprep
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def go_svm(ds,indices):
   X = np.arange(len(ds)).reshape(1, -1)
   Y = ds.values.reshape(1, -1)
   sc_X = skprep.StandardScaler()
   sc_y = skprep.StandardScaler()
   X = X.reshape(-1, 1)
   X = sc_X.fit_transform(X)
   y = sc_y.fit_transform(Y)

   regressor = SVR(kernel='rbf',C=250.0, gamma=20, epsilon=0.2)
   regressor.fit(X, y.flatten())

   X_grid = np.arange(len(ds))
   #X_grid = X_grid.reshape((len(X_grid), 1))

   ypred = regressor.predict(X)
   yfore = sc_y.inverse_transform([ypred])[0]

   plt.plot(ds.values, color='red', linewidth=3)
   plt.plot(yfore, color='blue')
   plt.xlabel('X')
   plt.ylabel('Y')
   plt.title("SVR")
   plt.show()

   return yfore