import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sklearn as sk
import matplotlib.pyplot as plt

def go_rf(ds,look_back=3,verbose=False):
   if(look_back!=3):
      print("ERROR, look_back must be 3 in this applicaition")
      return
   x = np.arange(len(ds))
   y = ds #.values
   x_train, x_test = x[:-look_back], x[-look_back:]
   y_train, y_test = y[:-look_back], y[-look_back:]
   x_train = np.vstack(x_train)
   y_train = np.vstack(y_train)
   x_test = np.vstack(x_test)
   y_test = np.vstack(y_test)
   model = rf = RandomForestRegressor(n_estimators=500)
   model.fit(x_train, y_train)
   pred = model.predict(x_test)
   mse = sk.metrics.mean_absolute_error(y_test, pred)
   print("MSE={}".format(mse))
   ypred = model.predict(x_train)
   yfore = np.zeros(look_back)
   # rolling forecast, valid only if look_back=3
   for i in np.arange(look_back):
      yfore[i] = model.predict(x_test)[0]
      x_test[0,0] = x_test[1,0]
      x_test[1,0] = x_test[2,0]
      x_test[2,0] = yfore[i]

   if verbose:
      plt.plot(y)
      plt.plot(ypred)
      plt.plot([None for x in ypred]+[x for x in yfore])
      plt.show()
   return yfore

   '''
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
   '''
