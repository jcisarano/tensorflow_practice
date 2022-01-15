import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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


def run():
    # plot_sigmoid_function()
    # explore_keras_initializers()
    # plot_leaky_relu()
    list_keras_activations()

