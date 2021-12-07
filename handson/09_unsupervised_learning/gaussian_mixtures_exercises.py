import numpy as np
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


def get_blob_data():
    x1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
    x1 = x1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
    x2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
    x2 = x2 + [6, -8]
    X = np.r_[x1, x2]
    y = np.r_[y1, y2]

    return X, y


def examine_gm(X, y):
    gm = GaussianMixture(n_components=3, n_init=10, random_state=42)
    gm.fit(X)

    print("GM weights:", gm.weights_)
    print("GM means:", gm.means_)
    print("GM covariances:", gm.covariances_)
    print("GM converged:", gm.converged_)
    print("GM num iterations:", gm.n_iter_)
    print("GM predictions:", gm.predict(X))
    print("GM pred probs:", gm.predict_proba(X))




def run():
    X, y = get_blob_data()
    examine_gm(X, y)


