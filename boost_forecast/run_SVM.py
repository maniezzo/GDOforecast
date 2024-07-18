import sklearn.preprocessing as skprep
import sklearn.metrics as skmetrics
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def go_svm(ds, look_back=3, verbose=False):
   print("init SVM")

   X_train = np.arange(len(ds)-look_back).reshape(1,-1)
   X_test  = np.arange(len(ds)-look_back,len(ds)).reshape(1,-1)
   X_fore  = np.arange(len(ds),len(ds)+look_back)
   y_train = ds[:len(ds)-look_back]
   y_test  = ds[len(ds)-look_back:]

   # sacling data
   scaler_in  = skprep.MinMaxScaler()  # for inputs
   scaler_out = skprep.MinMaxScaler()  # for outputs

   X_train = scaler_in.fit_transform(X_train.reshape(-1, 1))
   y_train = scaler_out.fit_transform(y_train.reshape(-1, 1))

   X_test = scaler_in.transform(X_test.reshape(-1,1))
   y_test = scaler_out.transform(y_test.reshape(-1,1))

   X_fore = scaler_in.transform(X_fore.reshape(-1, 1))

   param_grid = {"C": np.linspace(10 ** (-2), 10 ** 3, 100),
                 'gamma': np.linspace(0.0001, 1, 20)}

   '''
   mod = SVR(kernel='rbf',C=250.0, gamma=20, epsilon=0.2)
   model = GridSearchCV(estimator=mod, param_grid=param_grid,
                        scoring="neg_mean_squared_error", verbose=1)
   '''
   model = SVR(kernel='rbf',C=5.0, gamma=150.0, epsilon=0.1)
   model = model.fit(X_train, y_train.ravel())

   # prediction
   predicted_train = model.predict(X_train)
   predicted_test  = model.predict(X_test)
   forecasted      = model.predict(X_fore)

   # inverse_transform because prediction is done on scaled inputs
   predicted_train = scaler_out.inverse_transform(predicted_train.reshape(-1, 1))
   predicted_test  = scaler_out.inverse_transform(predicted_test.reshape(-1, 1))
   forecasted      = scaler_out.inverse_transform(forecasted.reshape(-1, 1))

   # plot
   real = np.concatenate((y_train, y_test))
   plt.plot(real, color='blue', label='Real')
   plt.plot(predicted_train, color='green', label='Model train')
   plt.plot(range(len(predicted_train),len(predicted_train)+len(predicted_test)), predicted_test, color='orange', label='Model test')
   plt.plot(range(len(ds),len(ds)+look_back), forecasted, color='red', label='forecast')
   plt.title('Prediction')
   plt.xlabel('Time')
   plt.legend()
   plt.show()

   # error
   forcast = np.concatenate((predicted_train, predicted_test))
   print("MSE: ", skmetrics.mean_squared_error(real, forcast), " R2: ", skmetrics.r2_score(real, forcast))
   #print(best_model.best_params_)

   return yfore