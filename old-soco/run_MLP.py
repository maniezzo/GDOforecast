import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential # pip install keras
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import math

# considers a vector of externals
def create_dataset(arrdata,indices,look_back=1):
	dataX, dataY = [], []
	for i in range(len(arrdata) - look_back):
		a = np.append(arrdata[i:(i + look_back)],indices[i + look_back])
		dataX.append(a)
		dataY.append(arrdata[i + look_back])
	return np.array(dataX), np.array(dataY)

def go_MLP(ds,indices,look_back = 12, lr=0.05, niter=1000):
	np.random.seed(550)                 # for reproducibility
	data = ds.values.reshape(-1, 1)     # time series values, 2D for compatibility with standardscaler
	data = data.astype('float32') # needed for MLP input
	externals = np.array([indices[i % 12] for i in np.arange(len(ds))]).reshape(-1, 1)
	numExternals = 1
	scaler  = StandardScaler()
	scaler.fit(data)
	dataset = scaler.transform(data)

	# split into train and test sets
	train_size  = int(len(dataset) - look_back)
	test_size   = len(dataset) - train_size
	train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
	print("Len train={0}, len test={1}".format(len(train), len(test)))

	# sliding window matrices (look_back = window width); dim = n - look_back - 1

	testdata = np.concatenate((train[-look_back:],test))
	trainX, trainY = create_dataset(train,externals[:len(train)], look_back)
	testX,  testY  = create_dataset(testdata,externals[-len(testdata):], look_back)

	# Multilayer Perceptron model
	loss_function = 'mean_squared_error'
	from tensorflow.keras.optimizers import Adam
	optimizer = Adam(learning_rate=lr)
	model = Sequential()
	nhid1 = look_back+numExternals
	nhid2 = look_back+numExternals
	nout = 1
	model.add(Dense(nhid1, input_dim=look_back+numExternals, activation='relu'))
	model.add(Dense(nhid2, activation='relu'))
	model.add(Dense(nout))
	model.compile(loss=loss_function, optimizer=optimizer)
	model.fit(trainX, trainY, epochs=niter, batch_size=10, verbose=0)

	# Estimate model performance
	trainScore = model.evaluate(trainX, trainY, verbose=0)
	print('Train Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(trainScore, math.sqrt(trainScore)))
	testScore = model.evaluate(testX, testY, verbose=0)
	print('Test Score: MSE: {0:0.3f} RMSE: ({1:0.3f})'.format(testScore, math.sqrt(testScore)))
	# generate predictions for training and forecast for plotting
	trainPredict = model.predict(trainX)
	testForecast = model.predict(testX)

	numpred=3
	yfore = np.zeros(numpred)
	indata = np.append(dataset[-look_back:],indices[len(dataset) % 12])
	for ii in np.arange(numpred):
		yfore[ii] = model.predict(np.asarray([indata]))
		for j in np.arange(1,look_back):
			indata[j-1]=indata[j]
		indata[look_back-1] = yfore[ii]
		i = ii + len(dataset) +1 # next index
		indata[look_back] = indices[i % 12]

	trainPredict = scaler.inverse_transform(trainPredict).flatten()
	testForecast = scaler.inverse_transform(testForecast).flatten()
	yfore        = scaler.inverse_transform(yfore.reshape(-1, 1)).flatten()

	plt.plot(ds.values)
	plt.plot(np.concatenate((np.full(look_back, np.nan), trainPredict[:])))
	plt.plot(np.concatenate((np.full(len(train), np.nan),testForecast[:])))
	plt.plot(np.concatenate((np.full(len(ds), np.nan),yfore[:])))
	plt.ylim(15,50)
	plt.show()

	return yfore
