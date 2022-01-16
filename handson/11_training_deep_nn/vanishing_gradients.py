import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from tensorflow import keras

def logit(z):
    return 1 / (1+np.exp(-z))


def plot_sigmoid_function():
    z = np.linspace(-5, 5, 200)

    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [1, 1], 'k--')
    plt.plot([0, 0], [-0.2, 1.2], 'k-')
    plt.plot([-5, 5], [-3/4, 7/4], 'g--')
    plt.plot(z, logit(z), 'b-', linewidth=2)

    props = dict(facecolor='black', shrink=0.1)
    plt.annotate('Saturating', xytext=(3.5, 0.7), xy=(5, 1), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Saturating', xytext=(-3.5, 0.3), xy=(-5, 0), arrowprops=props, fontsize=14, ha="center")
    plt.annotate('Linear', xytext=(2, 0.2), xy=(0, 0.5), arrowprops=props, fontsize=14, ha="center")

    plt.grid(True)
    plt.title("Sigmoid activation function")
    plt.axis([-5, 5, -0.2, 1.2])

    plt.show()


def explore_keras_initializers():
    # list built-in kernel initializers:
    print([name for name in dir(keras.initializers) if not name.startswith("_")])
    # use one via string name:
    print(keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal"))
    # use via api
    init = keras.initializers.VarianceScaling(scale=2., mode="fan_avg", distribution="uniform")
    print(keras.layers.Dense(10, activation="relu", kernel_initializer=init))


def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)


def plot_leaky_relu():
    z = np.linspace(-5, 5, 200)

    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.plot(z, leaky_relu(z, 0.05), 'b-', linewidth=2)

    props = dict(facecolor='black', shrink=0.1)
    plt.annotate("Leak", xytext=[-3.5, 0.5], xy=[-5, -0.2], arrowprops=props, fontsize=14, ha="center")
    plt.axis([-5, 5, -0.5, 4.2])
    plt.grid(True)
    plt.title("Leaky ReLU activation function")
    plt.show()


def list_keras_activations():
    print([m for m in dir(keras.activations) if not m.startswith("_")])
    print([m for m in dir(keras.layers) if "relu" in m.lower()])


def load_data():
    # load data into training, validation and test sets, normalize to 0-1 range
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
    X_train, X_valid = X_train_full[:55000]/255., X_train_full[55000:]/255.
    X_test = X_test / 255.
    y_train, y_valid = y_train_full[:55000], y_train_full[55000:]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train_fashion_mnist_prelu():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=10,
              validation_data=[X_valid, y_valid])


def elu(z, alpha=1):
    return np.where(z < 0, alpha*(np.exp(z)-1), z)


alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1/np.sqrt(2)) * np.exp(1/2) - 1)
scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (2 * erfc(np.sqrt(2))*np.e**2 + np.pi*erfc(1/np.sqrt(2))**2*np.e - 2*(2+np.pi)*erfc(1/np.sqrt(2))*np.sqrt(np.e)+np.pi+2)**(-1/2)


def selu(z, scale=scale_0_1, alpha=alpha_0_1):
    return scale * elu(z, alpha)


def plot_selu():
    z = np.linspace(-5, 5, 200)

    plt.plot([-5, 5], [0, 0], "k-")
    plt.plot([0, 0], [-5, 5], "k-")
    plt.plot([-5, 5], [-1.758, -1.758], "k--")
    plt.plot(z, selu(z), "b-", linewidth=2)

    plt.grid(True)
    plt.title("SELU activation function")
    plt.axis([-5, 5, -2.2, 3.2])
    plt.show()


def show_selu_hyperparams():
    np.random.seed(42)
    Z = np.random.normal(size=(500, 100))
    for layer in range(1000):
        W = np.random.normal(size=(100, 100), scale=np.sqrt(1/100))  # LeCun initialization
        Z = selu(np.dot(Z, W))
        means = np.mean(Z, axis=0).mean()
        stds = np.std(Z, axis=0).mean()
        if layer % 100 == 0:
            print("Layer {}: mean {:.2f}, std deviation {:.2f}".format(layer, means, stds))


def train_fashion_mnist_selu():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="selu",
                                 kernel_initializer="lecun_normal"))
    for layer in range(99):
        model.add(keras.layers.Dense(100,
                                     activation="selu",
                                     kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])

    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    model.fit(X_train_scaled, y_train, epochs=5,
              validation_data=(X_valid_scaled, y_valid))


def train_fashion_mnist_relu_scaled():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()
    np.random.seed(42)
    tf.random.set_seed(42)

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="relu", kernel_initializer="he_normal"))
    for layer in range(99):
        model.add(keras.layers.Dense(100, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizer.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])

    pixel_means = X_train.mean(axis=0, keepdims=True)
    pixel_stds = X_train.std(axis=0, keepdims=True)
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds

    model.fit(X_train_scaled, y_train, epochs=5,
              validation_data=(X_valid_scaled, y_valid))


def plot_elu():
    z = np.linspace(-5, 5, 200)

    plt.plot([-5, 5], [0, 0], "k-")
    plt.plot([-5, 5], [-1, -1], "k--")
    plt.plot([0, 0], [-5, 5], "k-")
    plt.plot(z, elu(z), "b-", linewidth=2)

    plt.grid(True)
    plt.title(r"ELU activation function($\alpha=1$)")
    plt.axis([-5, 5, -2.2, 3.2])
    plt.show()


def train_fashion_mnist_relu():
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_data()

    np.random.seed(42)
    tf.random.set_seed(42)

    # Uses Keras LeakyReLU built-in layers
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28], name="input_layer"),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=10,
              validation_data=(X_valid, y_valid))


def run():
    # plot_sigmoid_function()
    # explore_keras_initializers()
    # plot_leaky_relu()
    # list_keras_activations()
    # train_fashion_mnist_relu()
    # train_fashion_mnist_prelu()
    # plot_elu()
    # plot_selu()
    # show_selu_hyperparams()
    # train_fashion_mnist_selu()
    train_fashion_mnist_relu()

