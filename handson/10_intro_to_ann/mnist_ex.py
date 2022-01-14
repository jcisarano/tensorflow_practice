import tensorflow as tf
import numpy as np

def run():
    tf.random.set_seed(42)
    np.random.seed(42)

    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_valid = X_train_full[5000:] / 255., X_train_full[:5000] / 255.
    y_train, y_valid = y_train_full[5000:] / 255., y_train_full[:5000] / 255.
    X_test = X_test / 255.

    # print("X_train_full shape:", X_train_full.shape, "y_train_full shape:", y_train_full.shape)
    # print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    # print("X_valid shape:", X_valid.shape, "y_valid shape:", y_valid.shape)
    # print("X_test shape", X_test.shape)#
    # print(X_test[0])

    # print("mnist")
