import numpy as np
from sklearn.datasets import make_swiss_roll, fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.model_selection import train_test_split


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


    
