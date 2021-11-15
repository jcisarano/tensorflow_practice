import numpy as np
from sklearn.datasets import make_swiss_roll, fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def get_data(n_samples=1000, noise=0.2):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=41)
    return X, t

def get_mnist():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    mnist.target = mnist.target.astype(np.uint8)

    X = mnist["data"]
    y = mnist["target"]

    return X, y
    # return train_test_split(X, y)

def run():
    X, t = get_data()

    mds = MDS(n_components=2, random_state=42)
    X_reduced_mds = mds.fit_transform(X)

    isomap = Isomap(n_components=2)
    X_reduced_isomap = isomap.fit_transform(X)

    tsne = TSNE(n_components=2, random_state=42)
    X_reduced_tsne = tsne.fit_transform(X)

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_mnist, y_mnist = get_mnist()
    lda.fit(X_mnist, y_mnist)
    X_reduced_lda = lda.transform(X_mnist)

    titles = ["MDS", "Isomap", "t-SNE"]

    plt.figure(figsize=(11, 4))

    for subplot, title, X_reduced in zip((131, 132, 133), titles, (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
        plt.subplot(subplot)
        plt.title(title, fontsize=14)
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
        plt.xlabel("$z_1$", fontsize=18)
        if subplot == 131:
            plt.ylabel("$z_2$", fontsize=18, rotation=0)
        plt.grid(True)

    plt.show()
