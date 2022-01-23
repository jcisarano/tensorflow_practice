"""
Practice training a deep neural network on the CIFAR10 image dataset.
    a) Build a DNN with 20 hidden layers of 100 neurons each (that's too many, but it's the point of this exercise).
        Use He initialization and the ELU activation function.
    b) Using Nadam optimization and early stopping, train the network on the CIFAR10 dataset. You can load it with
        keras.datasets.cifar10.load_data(). The dataset is composed of 60,000 32 × 32–pixel color images (50,000 for
        raining, 10,000 for testing) with 10 classes, so you'll need a softmax output layer with 10 neurons. Remember
        to search for the right learning rate each time you change the model's architecture or hyperparameters.
    c) Now try adding Batch Normalization and compare the learning curves: Is it converging faster than before? Does it
        produce a better model? How does it affect training speed?
    d) Try replacing Batch Normalization with SELU, and make the necessary adjustements to ensure the network
        self-normalizes (i.e., standardize the input features, use LeCun normal initialization, make sure the DNN
        contains only a sequence of dense layers, etc.).
    e) Try regularizing the model with alpha dropout. Then, without retraining your model, see if you can achieve better
        accuracy using MC Dropout.
    f) Retrain your model using 1cycle scheduling and see if it improves training speed and model accuracy.
"""
import keras.models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def load_cfir10():
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train_full = X_train_full / 255.
    X_test = X_test / 255.
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def load_and_scale_cfir10():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_cfir10()
    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    return X_train_scaled, X_valid_scaled, X_test_scaled, y_train, y_valid, y_test


def get_class_names():
    return ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def create_model(n_classes, n_layers=20, n_neurons=100):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[32, 32]))
    for _ in range(n_layers):
        model.add(keras.layers.Dense(n_neurons, activation="elu", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Dense(n_classes, activation="softmax"))

    return model


def visualize_cfir10_samples(X, y):
    plt.figure(figsize=(7.2, 2.4))
    for index, image in enumerate(X):
        plt.subplot(5, 10, index+1)
        plt.imshow(image)
        plt.axis(False)
        # plt.title(y[index])
    # plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.show()


def run():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_cfir10()
    class_names = get_class_names()
    print(X_train.shape, X_valid.shape, X_test.shape)

    # visualize_cfir10_samples(X_train[:50], y_train)

    model = create_model(n_classes=len(class_names))

    print("example")
