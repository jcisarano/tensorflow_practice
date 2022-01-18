import tensorflow as tf
import numpy as np
from tensorflow import keras

from helper_functions import load_data

MODEL_A_PATH: str = "saved_models/model_a.h5"
MODEL_B_PATH: str = "saved_models/model_b.h5"


def split_data(X, y):
    y_5_or_6 = (y == 5) | (y == 6)
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # reduce indices above 6 by 2
    y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


def _split_data(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


def load_split_data():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    (X_train_A, y_train_A), (X_train_B, y_train_B) = split_data(X_train, y_train)
    (X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_data(X_valid, y_valid)
    (X_test_A, y_test_A), (X_test_B, y_test_B) = split_data(X_test, y_test)
    X_train_B = X_train_B[:200]
    y_train_B = y_train_B[:200]

    return X_train_A, X_valid_A, X_test_A, X_train_B, X_valid_B, X_test_B, y_train_A, y_valid_A, y_test_A, y_train_B, y_valid_B, y_test_B


def create_model(n_output=8, output_activation="softmax"):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model.add(keras.layers.Dense(n_hidden, activation="selu"))
    model.add(keras.layers.Dense(n_output, activation=output_activation))

    return model


def run():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train_A, X_valid_A, X_test_A, X_train_B, X_valid_B, X_test_B, y_train_A, y_valid_A, y_test_A, y_train_B, \
    y_valid_B, y_test_B = load_split_data()

    print(X_train_A.shape)
    print(X_train_B.shape)
    print(y_train_A[:30])
    print(y_train_B[:30])

    np.random.seed(42)
    tf.random.set_seed(42)

    if False:
        model_A = create_model()
        model_A.compile(loss="sparse_categorical_crossentropy",
                        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                        metrics=["accuracy"])
        model_A.fit(X_train_A, y_train_A, epochs=20,
                    validation_data=(X_valid_A, y_valid_A),
                    workers=-1)
        model_A.save(MODEL_A_PATH)

    if False:
        model_B = create_model(n_output=1, output_activation="sigmoid")
        model_B.compile(loss="binary_crossentropy",
                        optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                        metrics=["accuracy"])
        model_B.fit(X_train_B, y_train_B, epochs=20,
                    validation_data=(X_valid_B, y_valid_B),
                    workers=-1)
        model_B.save(MODEL_B_PATH)

    model_A = keras.models.load_model(MODEL_A_PATH)
    # print(model_A.summary())
    model_B = keras.models.load_model(MODEL_B_PATH)
    # print(model_B.summary())

    # copy a model into another, and add a different activation layer
    model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
    model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid", name="activation"))

    # clone the model before copying--this makes sure training one does not affect the other
    model_A_clone = keras.models.clone_model(model_A)
    model_A_clone.set_weights(model_A.get_weights())
    model_B_on_A = keras.models.Sequential(model_A_clone.layers[:-1])
    model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid", name="activation"))
    print(model_B_on_A.summary())



    print("reuse keras model")
