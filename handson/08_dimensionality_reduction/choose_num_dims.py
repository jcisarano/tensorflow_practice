"""
One way to choose the number of dimensions is to calculate the variance as dimensions are reduced, and keep the
variance above 95%.

But if you are reducing dimensions for visualization, you'll have to go to just 2 or 3 dims.
"""
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_mnist_train_test_split():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X = mnist["data"]
    y = mnist["target"]

    return train_test_split(X, y)


def pca_reduce_and_plot(X_train):
    # create the PCA, then fit it, then determine number of dims to keep variance above 95%
    pca = PCA()
    pca.fit(X_train)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    # print(cumsum.shape)
    d = np.argmax(cumsum >= 0.95) + 1
    print(d)

    # plot the variance curve
    plt.figure(figsize=(6, 4))
    plt.plot(cumsum, linewidth=3)
    plt.axis([0, 400, 0, 1])
    plt.xlabel("Dimensions")
    plt.ylabel("Explained Variance")
    plt.plot([d, d], [0, 0.95], "k:")
    plt.plot([0, d], [0.95, 0.95], "k:")
    plt.plot(d, 0.95, "ko")
    plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7), arrowprops=dict(arrowstyle="->"), fontsize=16)
    plt.grid(True)

    plt.show()


def pca_reduce(X_train):
    # or, specify the min variance when you create the PCA()
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_train)
    print(pca.n_components_)
    print(np.sum(pca.explained_variance_ratio_))
    


def run():
    X_train, X_test, y_train, y_test = get_mnist_train_test_split()

    # pca_reduce_and_plot(X_train)
    X_reduced = pca_reduce(X_train)





