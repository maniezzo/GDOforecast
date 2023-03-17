import sklearn.preprocessing as skprep

from sklearn.svm import SVR

sc_X = skprep.StandardScaler()
sc_y = skprep.StandardScaler()
X = X.reshape(-1, 1)
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(Y)

regressor = SVR(kernel='rbf',C=250.0, gamma=20, epsilon=0.2)
regressor.fit(X, y.flatten())

y_pred = regressor.predict([[6.5]])
y_pred = sc_y.inverse_transform(y_pred)