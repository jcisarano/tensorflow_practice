from tensorflow import keras


def load_data():
    # load data into training, validation and test sets, normalize to 0-1 range
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train, X_valid = X_train_full[:55000]/255., X_train_full[55000:]/255.
    X_test = X_test / 255.
    y_train, y_valid = y_train_full[:55000], y_train_full[55000:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test

