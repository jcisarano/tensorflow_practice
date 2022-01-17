import tensorflow as tf
import numpy as np
from tensorflow import keras

from helper_functions import load_data


def split_data_(X, y):
    y_5_or_6 = (y == 5) | (y == 6)
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # reduce indices above 6 by 2
    y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


def split_data(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
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


def run():
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train_A, X_valid_A, X_test_A, X_train_B, X_valid_B, X_test_B, y_train_A, y_valid_A, y_test_A, y_train_B, \
    y_valid_B, y_test_B = load_split_data()

    # print(X_train_A.shape)
    # print(X_train_B.shape)
    # print(y_train_A[:30])
    # print(y_train_B[:30])

    np.random.seed(42)
    tf.random.set_seed(42)

    model_A = keras.models.Sequential()
    model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
    for n_hidden in (300, 100, 50, 50, 50):
        model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
    model_A.add(keras.layers.Dense(8, activation="softmax"))

    model_A.compile(loss="sparse_categorical_crossentropy",
                    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                    metrics=["accuracy"])
    model_A.fit(X_train_A, y_train_A, epochs=20,
                validation_data=(X_valid_A, y_valid_A),
                workers=-1)
    model_A.save("saved_models/model_a.h5")


    print("reuse keras model")
