import sklearn.preprocessing as skprep
import sklearn.metrics as skmetrics
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

def go_svm(ds, look_back=3, verbose=False):
   train = np.array([[9.26,11.01],[11.01,22.72],[22.72,20.75],[20.75,11.54],[11.54,11.85],[11.85,18.17],[18.17,16.05],[16.05,17.98],[17.98,14.85],[14.85,12.62],[12.62,16.95],[16.95,16.81],[16.81,16.23],[16.23,21.81],[21.81,22.47],[22.47,20.37],[20.37,16.68],[16.68,17.07],[17.07,20.48],[20.48,21.99],[21.99,25.54],[25.54,21.1,],[21.1,16.91],[16.91,24.23],[24.23,27.37],[27.37,30.55],[30.55,28.47],[28.47,26.74],[26.74,40.37],[40.37,36.55],[36.55,39.65],[39.65,45.58],[45.58,48.91],[48.91,37.82],[37.82,39.7,],[39.7,36.09],[36.09,25.33],[25.33,23.64],[23.64,18.33],[18.33,21.59],[21.59,22.4,],[22.4,15.89],[15.89,18.94],[18.94,21.78],[21.78,19.38],[19.38,17.81],[17.81,21.33],[21.33,22.61],[22.61,27.11],[27.11,26.48],[26.48,19.87],[19.87,18.57],[18.57,14.03],[14.03,18.82],[18.82,22.46],[22.46,22.33],[22.33,21.58],[21.58,22.66],[22.66,19.51],[19.51,21.54],[21.54,20.58],[20.58,20.48]])
   test  = np.array([[20.48,25.78],[25.78,21.89],[21.89,19.61],[19.61,22.95],[22.95,21.67],[21.67,26.03],[26.03,21.96],[21.96,21.81],[21.81,21.91],[21.91,21.82],[21.82,19.6,],[19.6,24.61],[24.61,30.97],[30.97,18.29],[18.29,19.84],[19.84,20.81],[20.81,29.17],[29.17,24.01],[24.01,21.3,],[21.3,25.08],[25.08,27.18],[27.18,26.59],[26.59,25.99],[25.99,28.74],[28.74,25.32],[25.32,27.56],[27.56,28.69]])
   # sacling data
   scaler_in = skprep.MinMaxScaler()  # for inputs
   scaler_out = skprep.MinMaxScaler()  # for outputs

   X_train = scaler_in.fit_transform(train[:, 0].reshape(-1, 1))
   y_train = scaler_out.fit_transform(train[:, 1].reshape(-1, 1))

   X_test = scaler_in.transform(test[:, 0].reshape(-1, 1))
   y_test = scaler_out.transform(test[:, 1].reshape(-1, 1))

   param_grid = {"C": np.linspace(10 ** (-2), 10 ** 3, 100),
                 'gamma': np.linspace(0.0001, 1, 20)}

   mod = SVR(epsilon=0.1, kernel='rbf')
   model = GridSearchCV(estimator=mod, param_grid=param_grid,
                        scoring="neg_mean_squared_error", verbose=0)

   best_model = model.fit(X_train, y_train.ravel())

   # prediction
   predicted_train = model.predict(X_train)
   predicted_test = model.predict(X_test)

   # inverse_transform because prediction is done on scaled inputs
   predicted_train = scaler_out.inverse_transform(predicted_train.reshape(-1, 1))
   predicted_test = scaler_out.inverse_transform(predicted_test.reshape(-1, 1))

   # plot
   #forcast = np.concatenate((predicted_train, predicted_test))
   real = np.concatenate((train[:, 1], test[:, 1]))
   plt.plot(real, color='blue', label='Real')
   plt.plot(predicted_train, color='green', label='Model train')
   plt.plot(range(len(predicted_train),len(predicted_train)+len(predicted_test)), predicted_test, color='red', label='Model test')
   plt.title('Prediction')
   plt.xlabel('Time')
   plt.legend()
   plt.show()

   # error
   print("MSE: ", skmetrics.mse(real, forcast), " R2: ", skmetrics.r2_score(real, forcast))
   print(best_model.best_params_)


   scaler = skprep.StandardScaler()
   Xtrain = np.arange(len(ds)-look_back).reshape(1, -1)
   Xtrain = Xtrain.reshape(-1, 1)
   Xtrain1 = scaler.fit_transform(Xtrain)

   regressor = SVR(kernel='rbf',C=250.0, gamma=20, epsilon=0.2)
   regressor.fit(Xtrain1,Ytrain1.flatten())

   ypred1 = regressor.predict(Xtrain1)
   ypred  = sc_y.inverse_transform([ypred1])[0]

   Xtest  = np.arange(len(ds)-look_back,len(ds)).reshape(1, -1)
   Xtest  = Xtest.reshape(-1, 1)

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