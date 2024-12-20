import matplotlib.pyplot as plt
import numpy as np


def sigmoid(z):
    return 1/ (1+np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z-eps)) / (2*eps)


def heaviside(z):
    return (z >= 0).astype(z.dtype)


def mlp_xor(x1, x2, activation=heaviside):
    return activation(-activation(x1+x2-1.5) + activation(x1+x2-0.5)-0.5)


def run():
    z = np.linspace(-5, 5, 200)
    plt.figure(figsize=(11, 4))

    plt.subplot(121)
    plt.plot(z, np.sign(z), "r-", linewidth=1, label="Step")
    plt.plot(z, sigmoid(z), "g--", linewidth=2, label="Sigmoid")
    plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)
    plt.legend(loc="center right", fontsize=14)
    plt.title("Activation functions", fontsize=14)
    plt.axis([-5, 5, -1.2, 1.2])

    plt.subplot(122)
    plt.plot(z, derivative(np.sign, z), "r-", linewidth=1, label="Step")
    plt.plot(0, 0, "ro", markersize=5)
    plt.plot(0, 0, "rx", markersize=10)
    plt.plot(z, derivative(sigmoid, z), "g--", linewidth=2, label="Sigmoid")
    plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
    plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
    plt.grid(True)

    plt.title("Derivatives", fontsize=14)
    plt.axis([-5, 5, -0.2, 1.2])

    plt.show()

    x1s = np.linspace(-0.2, 1.2, 100)
    x2s = np.linspace(-0.2, 1.2, 100)
    x1, x2 = np.meshgrid(x1s, x2s)

    z1 = mlp_xor(x1, x2, activation=heaviside)
    z2 = mlp_xor(x1, x2, activation=sigmoid)

    plt.figure(figsize=(10, 4))

    plt.subplot(121)
    plt.contourf(x1, x2, z1)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: heaviside", fontsize=14)
    plt.grid(True)

    plt.subplot(122)
    plt.contourf(x1, x2, z2)
    plt.plot([0, 1], [0, 1], "gs", markersize=20)
    plt.plot([0, 1], [1, 0], "y^", markersize=20)
    plt.title("Activation function: sigmoid", fontsize=14)
    plt.show()


