from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


def get_moons():
    """
    Get train/test data using moons dataset
    :return:
    """
    X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def run():
    X_train, X_test, y_train, y_test = get_moons()
