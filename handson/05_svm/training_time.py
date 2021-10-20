# calculate training time
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_moons


def run():
    X, y = make_moons(n_samples=1000, noise=0.4, random_state=42)
    # plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    # plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    # plt.show()

    tol = 0.1
    tols = []
    times = []
    for i in range(10):
        svm_clf = SVC(kernel="poly", gamma=3, C=10, tol=tol, verbose=1)
        t1 = time.time()
        svm_clf.fit(X, y)
        t2 = time.time()
        times.append(t2-t1)
        tols.append(tol)
        print(i, tol, t2-t1)
        tol /= 10
    plt.semilogx(tols, times, "bo-")
    plt.xlabel("Tolerance", fontsize=16)
    plt.ylabel("Time (seconds)", fontsize=16)
    plt.grid(True)
    plt.show()