import numpy as np
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN


def create_moons(n_samples=1000, noise=0.05, random_state=42):
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def simple_example(X, y):
    dbscan = DBSCAN(eps=0.05, min_samples=5)
    dbscan.fit(X)

    print(dbscan.labels_[:10])
    print(len(dbscan.core_sample_indices_))
    print(dbscan.core_sample_indices_[:10])
    print(dbscan.components_[:3])
    print(np.unique(dbscan.labels_))

    dbscan2 = DBSCAN(eps=0.2)
    dbscan2.fit(X)

    


def run():
    X, y = create_moons()
    simple_example(X, y)
