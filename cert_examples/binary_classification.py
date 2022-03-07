import tensorflow as tf

from utils import generate_circles


def simple_binary_classification(X, y):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])


def run():
    X, y = generate_circles()

    tf.random.set_seed(42)
    simple_binary_classification(X, y)

    print("binary classification")
