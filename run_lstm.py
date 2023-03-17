import pandas as pd, numpy as np, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import math

def go_lstm(ds,indices,look_back = 12, lr=0.05, niter=1000):
   np.random.seed(550)  # for reproducibility
   datas = ds.values  # time series values, 2D for compatibility
   datas = datas.astype('float32')  # needed for MLP input
   externals = np.array([indices[i % 12] for i in np.arange(len(ds))]).reshape(-1, 1)
   numExternals = 1

   # train and test set
   train = datas[:-look_back]
   test  = datas[-look_back:]

   from sklearn.preprocessing import MinMaxScaler
   scaler = MinMaxScaler()
   scaler.fit_transform(train.reshape(-1, 1))
   scaled_train_data = scaler.transform(train.reshape(-1, 1))
   scaled_test_data  = scaler.transform(test.reshape(-1, 1))

   from keras.preprocessing.sequence import TimeseriesGenerator
   n_input = look_back; n_features = 1
   generator = TimeseriesGenerator(scaled_train_data, scaled_train_data, length=n_input, batch_size=1)

   lstm_model = Sequential()
   lstm_model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(n_input, n_features), dropout=0.05))
   lstm_model.add(LSTM(15, activation='relu')) # stacked LSTM
   lstm_model.add(Dense(1))
   lstm_model.compile(optimizer='adam', loss='mse')
   lstm_model.fit(generator,epochs=niter,verbose=1)
   lstm_model.summary()
   losses_lstm = lstm_model.history.history['loss']

   plt.figure()
   plt.xticks(np.arange(0,21,1))   # convergence trace
   plt.plot(range(len(losses_lstm)),losses_lstm);
   plt.show()

   # rolling day prediction (validation)
   lstm_predictions_scaled = list()
   batch = scaled_train_data[-n_input:]
   curbatch = batch.reshape((1, n_input, n_features)) # 1 dim more
   for i in range(len(test)):
      lstm_pred = lstm_model.predict(curbatch)[0]
      lstm_predictions_scaled.append(lstm_pred)
      curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1)

   # rolling day forecast
   lstm_forecasts_scaled = list()
   batch = scaled_train_data[-n_input:]
   curbatch = batch.reshape((1, n_input, n_features)) # 1 dim more
   for i in range(len(test)):
      lstm_pred = lstm_model.predict(curbatch)[0]
      lstm_forecasts_scaled.append(lstm_pred)
      curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1)

   lstm_prediction = scaler.inverse_transform(lstm_predictions_scaled)
   lstm_forecast   = scaler.inverse_transform(lstm_forecasts_scaled)
   yfore = np.transpose(lstm_forecast).squeeze()

   plt.plot(ds, label="ds")
   plt.plot(train,label='train')
   plt.plot([None for x in train]+[x for x in yfore], label='yfore')
   plt.ylim(15, 50)
   plt.legend()
   plt.show()

   return yfore