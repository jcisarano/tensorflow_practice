"""
PCA method that uses the kernel trick, which allows effecient processing of high-dimensional feature sets
"""
from sklearn.datasets import make_swiss_roll


def get_data(n_samples=1000, noise=0.2):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=42)
    return X, t


def run():
    X, t = get_data()