# Set up proper datasets for training and testing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from simple_model import plot_decision_boundary


def split(X, y, train_set_percent=0.8):
    train_index = np.int(len(X) * train_set_percent)
    train_x = X[:train_index]
    test_x = X[train_index:]
    train_y = y[:train_index]
    test_y = y[train_index:]
    return train_x, train_y, test_x, test_y


def run(X, y):
    # print(len(X))
    X_train, y_train, X_test, y_test = split(X, y)
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # now create a model to train/eval with train/test data
    """tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=25, workers=-1, verbose=0)

    # evaluate on test data
    print(model.evaluate(X_test, y_test))

    # plot decision boundaries on training and test data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X=X_train, y=y_train, do_show=False)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X=X_test, y=y_test)

    # plot history object
    # shows accuracy increase to nearly 1 and loss decrease to nearly 0
    # print(pd.DataFrame(history.history))
    pd.DataFrame(history.history).plot()
    plt.title("Loss curves")
    plt.show()"""

    # finding ideal learning rate
    # using learning rate callback to monitor lr during training
    """tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
    # create the learning rate callback
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

    # model with lr_scheduler
    history = model.fit(X_train, y_train,
                        epochs=100,
                        callbacks=[lr_scheduler],
                        workers=-1)

    # plot history
    pd.DataFrame(history.history).plot(figsize=(10, 7), xlabel="epochs")
    plt.show()

    # plot learning rate vs loss
    # create range of learning rate values
    lrs = 1e-4 * (10**(tf.range(100)/20))
    plt.figure(figsize=(10, 7))
    plt.semilogx(lrs, history.history["loss"])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.title("Learning rate vs loss")
    plt.show()"""

    # another model, using the optimal learning rate discovered above: 0.02
    tf.random.set_seed(42)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                  metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=20, workers=-1)

    # evaluate with test dataset
    # turns out its accuracy is a little lower than earlier model (which had 1.0 accuracy)
    print(model.evaluate(X_test, y_test))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X=X_train, y=y_train, do_show=False)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X=X_test, y=y_test)

    # More classification evaluation metrics:
    # Accuracy, Precision, Recall, F1-score, Confusion matrix, classification report (sklearn)

    # check accuracy of model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Model loss on the test set: {loss}")
    print(f"Model accuracy on the test set: {(accuracy*100):.2f}%")

    # confusion matrix
    from sklearn.metrics import confusion_matrix
    y_preds = model.predict(X_test)
    # sigmoid output is prediction probability (range 0-1), but x_test is 0/1 binary values
    # confusion_matrix requires the data to be the same,
    # so convert the predictions to binary format by rounding
    y_preds = tf.round(y_preds)
    print(confusion_matrix(y_test, y_preds))
