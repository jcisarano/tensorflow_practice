from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats
import numpy as np


def show_iris_clusters(data):
    x = data.data
    y = data.target
    # print(data.target_names)

    plt.figure(figsize=(9, 3.5))
    plt.subplot(121)
    plt.plot(x[y == 0, 2], x[y == 0, 3], "yo", label="Iris setosa")
    plt.plot(x[y == 1, 2], x[y == 1, 3], "bs", label="Iris veriscolor")
    plt.plot(x[y == 2, 2], x[y == 2, 3], "g^", label="Iris virginica")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(fontsize=12)

    plt.subplot(122)
    plt.scatter(x[:, 2], x[:, 3], c="k", marker=".")
    plt.xlabel("Petal length", fontsize=14)
    plt.tick_params(labelleft=False)
    plt.show()


def predict_iris(iris_data):
    X = iris_data.data
    y = iris_data.target

    y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
    mapping = {}
    for class_id in np.unique(y):
        mode, _ = stats.mode(y_pred[y == class_id])
        mapping[mode[0]] = class_id
    print(mapping)
    y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])

    plt.figure(figsize=(7, 5))
    plt.plot(X[y_pred == 0, 2], X[y_pred == 0, 3], "yo", label="Cluster 1")
    plt.plot(X[y_pred == 1, 2], X[y_pred == 1, 3], "bs", label="Cluster 2")
    plt.plot(X[y_pred == 2, 2], X[y_pred == 2, 3], "g^", label="Cluster 3")
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=12)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = load_iris()
    # show_iris_clusters(data)
    predict_iris(data)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
