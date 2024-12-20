import tensorflow as tf
import numpy as np
from tensorflow import keras

from helper_functions import load_data


def batch_norm_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def batch_norm_model_separate_activation():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(300, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(100, use_bias=False),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dense(10, activation="softmax")
    ])

    return model


def run():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    # model = batch_norm_model()
    model = batch_norm_model_separate_activation()

    # print(model.summary())
    # bn1 = model.layers[1]
    # print([(var.name, var.trainable) for var in bn1.variables])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10,
              validation_data=(X_valid, y_valid),
              workers=1)

    print("bn")
