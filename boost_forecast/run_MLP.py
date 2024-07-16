import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
import copy

import torch
import tqdm
import torch.nn as nn           # neural network

def create_dataset(arrdata,look_back=3):
	dataX, dataY = [], []
	for i in range(len(arrdata) - look_back):
		a = arrdata[i:(i + look_back)]
		dataX.append(a)
		dataY.append(arrdata[i + look_back])
	return np.array(dataX), np.array(dataY)

def go_MLP(ds,look_back = 3, lr=0.05, niter=1000, verbose=False):
	np.random.seed(550)          # for reproducibility
	data = ds.reshape(-1, 1)     # time series values, 2D for compatibility
	scaler   = preprocessing.StandardScaler()
	X_scaled = scaler.fit_transform(data).T[0]

	# split into train and test sets
	train_size  = int(len(X_scaled) - look_back)
	test_size   = len(X_scaled) - train_size
	n_test_samples = test_size
	# sliding window matrices (look_back = window width); dim = n - look_back - 1
	X_train, y_train = create_dataset(X_scaled[:train_size],look_back)
	X_test,  y_test  = create_dataset(X_scaled[train_size-look_back:],look_back)
	print("Len train={0}, len test={1}".format(len(X_train), len(X_test)))

	# numpy array to tensor
	X_train = torch.FloatTensor(X_train)
	X_test  = torch.FloatTensor(X_test)
	y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1) # vuole così pytorch (!)
	y_test  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1) # vuole così pytorch (!)

	# Multilayer Perceptron model, one hidden layer
	model = nn.Sequential(
		nn.Linear(3, 3),
		nn.ReLU(),
		nn.Linear(3, 3),
		nn.ReLU(),
		nn.Linear(3, 1)
	)
	loss_fn = nn.MSELoss()  # mean square error
	optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
	n_epochs = 100  # number of epochs to run
	batch_size = 10  # size of each batch
	batch_start = torch.arange(0, len(X_train), batch_size)
	losses = []
	# keep the best model
	best_loss = np.inf
	best_weights = None
	for i in range(n_epochs):
		model.train()
		for start in batch_start:
			# take a batch
			X_batch = X_train[start:start + batch_size]
			y_batch = y_train[start:start + batch_size]
			# forward pass
			y_pred = model(X_batch)
			loss = loss_fn(y_pred, y_batch)
			# backward pass
			optimizer.zero_grad()
			loss.backward()
			# update weights
			optimizer.step()
		# evaluate accuracy at end of each epoch
		model.eval()
		y_pred = model(X_test)
		loss = loss_fn(y_pred, y_test)
		losses.append(loss.item())
		loss = float(loss)
		if loss < best_loss:
		   best_loss = loss
		   best_weights = copy.deepcopy(model.state_dict())

	# restore model and return best accuracy
	model.load_state_dict(best_weights)
	print("MSE: %.2f" % best_loss)
	print("RMSE: %.2f" % np.sqrt(best_loss))
	la = np.array(losses)
	plt.figure()
	plt.plot(la)
	plt.ylabel('Loss')
	plt.xlabel('epoch');
	plt.title("Losses")
	plt.show()

	numpred=3
	yfore = np.zeros(numpred)
	indata = X_scaled[-look_back:]
	X_fore = torch.FloatTensor(indata)
	for ii in np.arange(numpred):
		yfore[ii] = model(X_fore)
		for j in np.arange(1,look_back):
			indata[j-1]=indata[j]
		indata[look_back-1] = yfore[ii]
		X_fore = torch.FloatTensor(indata)

	trainPredict = scaler.inverse_transform(X_scaled[:train_size].reshape(-1, 1))
	testForecast = scaler.inverse_transform(X_scaled[train_size-look_back:].reshape(-1, 1))
	yfore        = scaler.inverse_transform(yfore.reshape(-1, 1)).flatten()

	plt.plot(ds)
	plt.plot(trainPredict.flatten(),label="train")
	tpq = len(trainPredict.flatten())-look_back
	plt.plot(range(tpq,tpq+len(testForecast.flatten() )), testForecast.flatten(),label="test")
	plt.plot(range(len(ds)-look_back, len(ds)-look_back+len(yfore)),yfore[:],label="forecast")
	plt.title("MLP model")
	plt.legend()
	plt.show()

	return yfore
