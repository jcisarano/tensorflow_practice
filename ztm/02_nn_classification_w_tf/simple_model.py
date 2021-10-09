# simple starting model

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Steps to build NN model
# 1. Prepare the data
# 2. Build the model: inputs, outputs, layers
# 3. Compile the model: loss function, optimizer, metrics
# 4. Fit the model to the training data
# 5. Evaluate and improve through experimentation


def run(X, y):
    """
    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(1)
    )
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["accuracy"])
    model.fit(X, y, epochs=5)

    model.fit(X, y, epochs=200, verbose=0)
    print(model.evaluate(X, y))
    """

    tf.random.set_seed(42)
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1),
    ])

    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
    model_1.fit(X, y, epochs=100, verbose=0)
    model_1.evaluate(X,y)

