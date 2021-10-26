import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import reciprocal, uniform

import visualization as vv
import matplotlib.pyplot as plt


def run():
    X, y = make_moons(n_samples=10000, noise=0.4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape)

    tree_model = DecisionTreeClassifier(random_state=42)
    tree_model.fit(X, y)

    y_pred = tree_model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print(np.sqrt(mse))

    tree_grid_search = DecisionTreeClassifier()
    params = {"max_depth": [1, 20]}
    grid_search = GridSearchCV(tree_grid_search, params, verbose=2, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    y_pred = grid_search.best_estimator_.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    print("GS MSE:", mse)
    print("GS RMSE:", np.sqrt(mse))


