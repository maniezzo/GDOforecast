
# random forest for making predictions for regression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
# define dataset
X, y = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=2)
# define the model
model = RandomForestRegressor()
# fit the model on the whole dataset
model.fit(X, y)
# make a single prediction
row = [[-0.89483109,-1.0670149,-0.25448694,-0.53850126,0.21082105,1.37435592,0.71203659,0.73093031,-1.25878104,-2.01656886,0.51906798,0.62767387,0.96250155,1.31410617,-1.25527295,-0.85079036,0.24129757,-0.17571721,-1.11454339,0.36268268]]
yhat = model.predict(row)
print('Prediction: %d' % yhat[0])

# forecast, train / test split
x = dataset.iloc[:, 1:]
y = dataset.iloc[:, 0]
x_train, x_valid = x[:-12], x[-12:]
y_train, y_valid = y[:-12], y[-12:]
mdl = rf = RandomForestRegressor(n_estimators=500)
mdl.fit(x_train, y_train)
pred = mdl.predict(x_valid)
mse = mean_absolute_error(y_valid, pred)
print("MSE={}".format(mse))

# Stats about the trees in random forest
n_nodes = []
max_depths = []
for ind_tree in model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)
print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

# plot first tree (index 0)
from sklearn.tree import plot_tree
fig = plt.figure(figsize=(15, 10))
plot_tree(model.estimators_[0],
          max_depth=2,
          feature_names=dataset.columns[:-1],
          class_names=dataset.columns[-1],
          filled=True, impurity=True,
          rounded=True)
plt.show()
