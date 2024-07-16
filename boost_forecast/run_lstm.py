import pandas as pd, numpy as np, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

class LSTMmodel(nn.Module):
   def __init__(self):
      super().__init__()
      nhidden = 40
      self.lstm = nn.LSTM(input_size=3, hidden_size=nhidden, num_layers=1, batch_first=True)
      self.linear = nn.Linear(nhidden, 1)

   def forward(self, x):
      x, _ = self.lstm(x)
      x = self.linear(x)
      return x

def create_dataset(dataset, lookback):
   X, y = [], []
   for i in range(len(dataset) - lookback):
      feature = dataset[i:i + lookback]
      target  = dataset[i + 1:i + lookback + 1] # due estremi avanti di 1
      X.append(feature)
      y.append(target)
   return torch.tensor(X), torch.tensor(y)

def go_lstm(ds, look_back = 12, lr=0.05, niter=1000, verbose=False):
   np.random.seed(550)  # for reproducibility
   datas = ds.astype('float32')  # needed for nn input

   # train and test set
   train = datas[:-2*look_back]
   test  = datas[-2*look_back:]

   X_train, y_train = create_dataset(train, lookback=look_back)
   X_test,  y_test  = create_dataset(test, lookback=look_back)

   model     = LSTMmodel()
   optimizer = optim.Adam(model.parameters())
   loss_fn   = nn.MSELoss()
   loader    = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

   n_epochs = 2000
   for epoch in range(n_epochs):
      model.train()
      for X_batch, y_batch in loader:
         y_pred = model(X_batch)
         loss = loss_fn(y_pred, y_batch)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
      # output trace
      if epoch % 100 == 0:
         model.eval()
         with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
         print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

   with torch.no_grad():
      # shift train predictions for plotting
      train_plot = np.ones_like(datas) * np.nan
      y_pred = model(X_train)
      y_pred = y_pred[:, -1]
      train_plot[look_back:len(train)] = model(X_train)[:, -1]
      # shift test predictions for plotting
      test_plot = np.ones_like(datas) * np.nan
      test_plot[len(train) + look_back:len(datas)] = model(X_test)[:, -1]
   # plot
   plt.plot(datas)
   plt.plot(train_plot, c='r')
   plt.plot(test_plot,  c='g')
   plt.show()

   # rolling day prediction (validation)
   lstm_predictions_scaled = list()
   batch = scaled_train[-n_input:]
   curbatch = batch.reshape((1, n_input, n_features)) # 3D, adding num of
   for i in range(len(test)):
      lstm_pred = lstm_model.predict(curbatch)[0] # one dim less to fit inverse scaling later
      lstm_predictions_scaled.append(lstm_pred)
      curbatch = np.append(curbatch[:,1:,:],[[lstm_pred]],axis=1) # mind the dimensions

   # rolling day forecast
   lstm_forecasts_scaled = list()
   scaled_datas = scaler.transform(datas.reshape(-1, 1))
   batch = scaled_datas[-n_input:]
   curbatch = batch.reshape((1, n_input, n_features)) # 1 dim more
   for i in range(len(test)):
      lstm_fore = lstm_model.predict(curbatch)[0]
      lstm_forecasts_scaled.append(lstm_fore)
      curbatch = np.append(curbatch[:,1:,:],[[lstm_fore]],axis=1)

   lstm_prediction = scaler.inverse_transform(lstm_predictions_scaled)
   lstm_forecast   = scaler.inverse_transform(lstm_forecasts_scaled)
   ypred = np.transpose(lstm_prediction).squeeze()
   yfore = np.transpose(lstm_forecast).squeeze()

   plt.plot(ds, label="ds")
   plt.plot(train,label='train')
   plt.plot([None for x in train]+[x for x in ypred], label='predict')
   plt.plot([None for x in ds]+[x for x in yfore], label='yfore')
   plt.ylim(15, 50)
   plt.legend()
   plt.show()

   return yfore