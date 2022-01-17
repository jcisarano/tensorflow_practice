from tensorflow import keras


def load_data():
    # load data into training, validation and test sets, normalize to 0-1 range
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train_full = X_train_full / 255.
    X_test = X_test / 255.
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


