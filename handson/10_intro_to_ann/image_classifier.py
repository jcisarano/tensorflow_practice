import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def visualize_data(X_train, y_train, X_valid, y_valid, X_test, y_test, class_names):
    # view an image
    plt.imshow(X_train[0], cmap="binary")
    plt.axis("off")
    plt.show()

    print(y_train)
    print("X_valid shape:", X_valid.shape)
    print("X_test shape:", X_test.shape)

    # view a bunch of dataset samples:
    n_rows = 4
    n_cols = 10
    plt.figure(figsize=(n_cols*1.2, n_rows*1.2))
    for row in range(n_rows):
        for col in range(n_cols):
            index = n_cols * row + col
            plt.subplot(n_rows, n_cols, index+1)
            plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
            plt.axis("off")
            plt.title(class_names[y_train[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


def visualize_model_details(model):
    print(model.layers)
    print(model.summary())

    keras.utils.plot_model(model, "mnist_fashion_model.png", show_shapes=True, show_dtype=True)
    hidden1 = model.layers[1]
    print(hidden1.name)
    print(model.get_layer(hidden1.name) is hidden1)
    weights, biases = hidden1.get_weights()
    print(weights)
    print(weights.shape)
    print(biases)
    print(biases.shape)


def run():
    print("TF version:", tf.__version__)
    print("Keras version:", keras.__version__)

    # load the keras built in fashion MNIST dataset
    fashion_mnist = keras.datasets.fashion_mnist
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
    print(X_train_full.shape)
    print(X_train_full.dtype)


    # split the training set into training and validation set.
    # scale pixel intensities to floats in 0-1 range
    X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
    X_test = X_test / 255.

    # class names array
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # visualize_data(X_train, y_train, X_valid, y_valid, X_test, y_test, class_names)

    # one way to build a Sequential model#
    # model = keras.models.Sequential()
    # model.add(keras.layers.Flatten(input_shape=[28, 28]))
    # model.add(keras.layers.Dense(300, activation="relu"))
    # model.add(keras.layers.Dense(100, activation="relu"))
    # model.add(keras.layers.Dense(10, activation="softmax"))
    # print(model.layers)
    # print(model.summary())

    keras.backend.clear_session()
    np.random.seed(42)
    tf.random.set_seed(432)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ])

    # visualize_model_details(model)

    # two equivalent ways to compile the model
    # model.compile(loss="sparse_categorical_crossentropy",
    #               optimizer="sgd",
    #               metrics=["accuracy"])
    model.compile(loss=keras.losses.sparse_categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(),
                  metrics=[keras.metrics.sparse_categorical_accuracy])

    history = model.fit(X_train,
                        y_train,
                        epochs=30,
                        validation_data=(X_valid, y_valid),
                        workers=-1)
    print(history.params)
    print(history.epoch)
    print(history.history.keys())

    # plot loss curves
    import pandas as pd
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    model.evaluate(X_test, y_test)
    X_new = X_test[:3]
    y_prob = model.predict(X_new)
    print(y_prob.round(2))

    y_pred = np.argmax(model.predict(X_new), axis=-1)
    print(y_pred)
    print(np.array(class_names)[y_pred])

    y_new = y_test[:3]
    print(y_new)

    # visualize image predictions
    plt.figure(figsize=(7.2, 2.4))
    for index, image in enumerate(X_new):
        plt.subplot(1, 3, index+1)
        plt.imshow(image, cmap="binary", interpolation="nearest")
        plt.axis("off")
        plt.title(class_names[y_test[index]], fontsize=12)
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()




