# Exercise: train an SVM regressor on the California housing dataset.
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR
import matplotlib.pyplot as plt

from scipy.stats import reciprocal, uniform
from sklearn.model_selection import RandomizedSearchCV


def create_svm_linear_regressor(X, y, epsilon=1.5):
    svm_reg = LinearSVR(epsilon=epsilon, random_state=42)
    svm_reg.fit(X, y)
    return svm_reg


def plot_regressor(regressor, X, y):
    x1s = np.linspace(-1, 1, 800).reshape(100, 8)
    plt.figure(figsize=(10, 7))
    plt.scatter(X, y)
    y_pred = regressor.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.show()


def run():
    california = fetch_california_housing(as_frame=False)
    X = california["data"]
    y = california["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    reg = create_svm_linear_regressor(X=X_train_scaled, y=y_train, epsilon=0.0)
    y_pred = reg.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print("LinearSVC RMSE:", mse)
    print("LinearSVC RMSE:", np.sqrt(mse))

    rbf_reg = SVR(kernel="rbf", gamma="scale")
    params = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(rbf_reg, params, n_iter=10, verbose=2, cv=3, n_jobs=-1, random_state=42)
    rnd_search_cv.fit(X_train_scaled, y_train)
    y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    print("SVC rbf RMSE:", mse)
    print("SVC rbf RMSE:", np.sqrt(mse))

    # plot_regressor(reg, med, y_train)





