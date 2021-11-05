from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def get_moons():
    X, y = get_raw_moons()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, y_train, y_test


def get_raw_moons():
    return make_moons(n_samples=500, noise=0.30, random_state=42)