# moon data classifier

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import simple_model as sm


def examine_data(X, y):
    # Check out features and labels:
    print(X[:10])
    print(y[:10])

    pd_data = pd.DataFrame({"X0:": X[:, 0], "X1": X[:, 1], "label": y})
    print(pd_data)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.show()

    print(X.shape, y.shape)


def run():
    X, y = make_moons(n_samples=1000, noise=0.03, random_state=42)
    examine_data(X, y)

    tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    model.fit(X, y, epochs=250, workers=-1)

    sm.plot_decision_boundary(model, X, y)
