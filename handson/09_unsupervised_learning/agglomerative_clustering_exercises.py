import numpy as np
from sklearn.cluster import AgglomerativeClustering


def learned_params(estimator):
    return [attrib for attrib in dir(estimator)
            if attrib.endswith("_") and not attrib.startswith("_")]


def run():
    X = np.array([0, 2, 5, 9.5]).reshape(-1, 1)
    agg = AgglomerativeClustering(linkage="complete").fit(X)
    print(learned_params(agg))

    print(agg.children_)

