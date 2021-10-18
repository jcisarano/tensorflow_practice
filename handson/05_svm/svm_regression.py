import matplotlib.pyplot as plt
import numpy as np
# from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.pipeline import Pipeline

import utils


def create_linear_svr(X, y):
    svm_reg = LinearSVR(epsilon=1.5, random_state=42)
    svm_reg.fit(X, y)
    return svm_reg


def run():
    np.random.seed(42)
    m = 50
    X = 2*np.random.rand(m, 1)  # shape (50,1)
    y = (4 + 3*X + np.random.randn(m, 1)).ravel()  # shape(50,)
    print(X)
    print(y)

    model_svmr = create_linear_svr(X, y)
    print(model_svmr)