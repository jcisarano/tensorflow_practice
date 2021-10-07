import numpy as np
from sklearn import  datasets



def run():
    iris = datasets.load_iris()

    np.random.seed(42)
    X = iris["data"][:, (2, 3)]  # petal length and width
    y = iris["target"]
    X_b = np.c_[np.ones((len(X), 1)), X]

    train_split_index = int(len(X_b) * 0.6)
    test_split_index = int(len(X_b) * 0.8)
    shuffled_indices = np.arange(len(X_b))
    np.random.shuffle(shuffled_indices)
    X_b_shuffle = X_b[shuffled_indices]
    y_shuffle = y[shuffled_indices]

    X_train = X_b_shuffle[:train_split_index]
    X_validation = X_b_shuffle[train_split_index:test_split_index]
    X_test = X_b_shuffle[test_split_index:]
    y_train = y_shuffle[:train_split_index]
    y_validation = y_shuffle[train_split_index:test_split_index]
    y_test = y_shuffle[test_split_index:]

    print(len(X_train), len(X_validation), len(X_test))
    print(len(y_train), len(y_validation), len(y_test))