
from sklearn.datasets import make_circles


def generate_circles(n_samples=1000):
    X, y = make_circles(n_samples,
                        noise=0.1,
                        random_state=42,
                        factor=0.2)
    return X, y