import pandas as pd, numpy as np, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tudata

class LSTMmodel(nn.Module):
   def __init__(self):
      super().__init__()
      nhidden = 40
      # batch_first=True: batch is created by sampling on the first tensor dimension, data records
      self.lstm = nn.LSTM(input_size=3, hidden_size=nhidden, num_layers=1, batch_first=True) # lstm layer
      self.linear = nn.Linear(nhidden, 1) # fully connected, normal layer

   def forward(self, x):
      x,_ = self.lstm(x)   # second element is the LSTM cells memory
      x   = self.linear(x)
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
   ds2    = ds.reshape(-1, 1).astype('float32')  # time series values, 2D for compatibility
   scaler = preprocessing.StandardScaler()
   datas  = scaler.fit_transform(ds2).T[0]

   # train and test set
   train = datas[:-2*look_back]
   test  = datas[-2*look_back:]

   X_train, y_train = create_dataset(train, lookback=look_back)
   X_test,  y_test  = create_dataset(test,  lookback=look_back)

   model     = LSTMmodel()
   optimizer = optim.Adam(model.parameters())
   loss_fn   = nn.MSELoss()
   loader    = tudata.DataLoader(tudata.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

   n_epochs = 1500
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
            ytest = model(X_test)
            test_rmse = np.sqrt(loss_fn(ytest, y_test))
         print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

   with torch.no_grad():
      # shift train predictions for plotting
      train_plot = np.ones_like(datas) * np.nan
      y_pred = model(X_train)
      y_pred = y_pred[:, -1]
      y_test = model(X_test)[:, -1]
      train_plot[look_back:len(train)] = model(X_train)[:, -1]
      # shift test predictions for plotting
      test_plot = np.ones_like(datas) * np.nan
      test_plot[len(train) + look_back:len(datas)] = model(X_test)[:, -1]

   # results, plot
   if verbose:
      plt.plot(datas)
      plt.plot(train_plot, c='r')
      plt.plot(test_plot,  c='g')
      plt.title("LSTM series")
      plt.show()

   numpred = 3
   yfore = np.zeros(numpred)
   indata = X_train[-look_back:]
   X_fore = torch.FloatTensor(indata)
   for ii in np.arange(numpred):
      outTensor = model(X_fore)
      yfore[ii] = outTensor[0,-1]
      for j in np.arange(1, look_back):
         indata[j - 1] = indata[j]
      indata[look_back - 1] = yfore[ii]
      X_fore = torch.FloatTensor(indata)

   train_size = len(X_train)
   trainPredict = scaler.inverse_transform(y_pred.reshape(-1, 1))
   testForecast = scaler.inverse_transform(y_test.reshape(-1, 1))
   yfore = scaler.inverse_transform(yfore.reshape(-1, 1)).flatten()

   if verbose:
      plt.plot(ds, 'g', label="ds")
      plt.plot(range( look_back-1, look_back-1+len(trainPredict)),trainPredict, label="train")
      tpq =  look_back +len(trainPredict.flatten())
      plt.plot(range(tpq, tpq + len(testForecast.flatten())), testForecast.flatten(),  label="test")
      plt.plot(range(len(ds) - look_back, len(ds) - look_back + len(yfore)), yfore[:],'r', label="forecast")
      plt.title("LSTM model")
      plt.legend()
      plt.show()

   return yfore