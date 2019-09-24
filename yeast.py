import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, linear_model, neural_network
from sklearn.metrics import accuracy_score,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("yeast.csv",names=['id',
'mcg','gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'location'], sep=';')
df.drop(['id'], 1, inplace=True)

corr = df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

X = np.array(df.drop(['location'],1))
y = np.array(df['location'])


# Regresion logistica
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
y, test_size = 0.2)

clf = linear_model.LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

conf = confusion_matrix(y_test, y_pred)
print(conf)

score = accuracy_score(y_test, y_pred, normalize=True)
print(score)

penalty = ['none', 'l2']
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
class_weight = ['balanced', None]
solver = ['saga', 'lbfgs']

param_grid = dict(penalty=penalty,
                  C=C,
                  class_weight=class_weight,
                  solver=solver)

grid = model_selection.GridSearchCV(estimator=clf,
                    param_grid=param_grid,
                    verbose=1,
                    n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('LR Best Score: ', grid_result.best_score_)
print('LR Best Params: ', grid_result.best_params_)


# Red neuronal
mlp = neural_network.MLPClassifier()
mlp.fit(X_train,y_train)
y_pred = mlp.predict(X_test)

conf = confusion_matrix(y_test, y_pred)
print(conf)

score = accuracy_score(y_test, y_pred, normalize=True)
print(score)

parameters = {'solver': ['lbfgs'],
              'max_iter': [1000,4000,7000,10000],
              'alpha': 10.0 ** -np.arange(1, 10),
              'hidden_layer_sizes':np.arange(10, 15),
              'random_state':[0,2,4,6,8,10]}

grid = model_selection.GridSearchCV(estimator=mlp,
                    param_grid=parameters,
                    verbose=1,
                    n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

print('MLPC Best Score: ', grid_result.best_score_)
print('MLPC Best Params: ', grid_result.best_params_)