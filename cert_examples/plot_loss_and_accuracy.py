import tensorflow as tf
from keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def train_model():
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", ]
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train.shape, y_train.shape)

    X_train_norm = X_train / X_train.max()
    X_test_norm = X_test / X_test.max()

    y_train_one_hot = tf.one_hot(y_train, depth=10)

    img_shape = (28, 28)
    tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=img_shape),
        tf.keras.layers.Dense(40, activation="relu"),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    return model.fit(X_train_norm, y_train_one_hot, epochs=10, workers=-1)


def run():
    history = train_model()
    print(history.history.keys())

    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.title("Model loss and accuracy")
    plt.legend(loc="center right")
    plt.show()

    pd.DataFrame(history.history).plot(title="Pandas plot same model loss and accuracy")
    plt.show()



    print("plot loss and accuracy")