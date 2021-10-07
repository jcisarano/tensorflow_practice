import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets


def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    y_one_hot = np.zeros((m, n_classes))
    y_one_hot[np.arange(m), y] = 1
    # print(y_one_hot[:10])
    return y_one_hot


def softmax(logits):
    exp = np.exp(logits)
    sum_exp = np.sum(exp, axis=1, keepdims=True)
    return exp / sum_exp


def run():
    iris = datasets.load_iris()

    np.random.seed(2042)
    X = iris["data"][:, (2, 3)]  # petal length and width
    y = iris["target"]
    X_b = np.c_[np.ones((len(X), 1)), X]

    train_split_index = int(len(X_b) * 0.6)
    test_split_index = int(len(X_b) * 0.8)
    shuffled_indices = np.arange(len(X_b))
    np.random.shuffle(shuffled_indices)
    X_b_shuffle = X_b[shuffled_indices]
    y_shuffle = y[shuffled_indices]

    X_train = X_b_shuffle[:train_split_index]
    X_validation = X_b_shuffle[train_split_index:test_split_index]
    X_test = X_b_shuffle[test_split_index:]
    y_train = y_shuffle[:train_split_index]
    y_validation = y_shuffle[train_split_index:test_split_index]
    y_test = y_shuffle[test_split_index:]
    # print(len(X_train), len(X_validation), len(X_test))
    # print(len(y_train), len(y_validation), len(y_test))

    y_train_one_hot = to_one_hot(y_train)
    y_validation_one_hot = to_one_hot(y_validation)
    y_test_one_hot = to_one_hot(y_test)

    print(y_train[:10])
    print(y_train_one_hot[:10])

    n_inputs = X_train.shape[1]  # == 3 (petal width, petal length, and bias term)
    n_outputs = len(np.unique(y_train))  # == 3 (three types of iris)

    # training
    eta = 0.01
    n_iterations = 5001
    m = len(X_train)
    epsilon = 1e-7

    Theta = np.random.randn(n_inputs, n_outputs)

    for iteration in range(n_iterations):
        logits = X_train.dot(Theta)
        y_proba = softmax(logits)
        loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon), axis=1))
        if iteration % 500 == 0:
            print(iteration, loss)
        error = y_proba - y_train_one_hot
        gradients = 1/m * X_train.T.dot(error)
        Theta = Theta - eta * gradients

    # print(Theta)
    logits = X_validation.dot(Theta)
    y_proba = softmax(logits)
    y_predict = np.argmax(y_proba, axis=1)
    accuracy_score = np.mean(y_predict == y_validation)
    print(accuracy_score)

    # same training function, but with l2 regularization and faster learning rate
    eta = 0.1
    n_iterations = 5001
    m = len(X_train)
    epsilon = 1e-7
    alpha = 0.1  # regularization hyperparameter

    Theta = np.random.randn(n_inputs, n_outputs)

    for iteration in range(n_iterations):
        logits = X_train.dot(Theta)
        y_proba = softmax(logits)
        xentropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon), axis=1))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
        loss = xentropy_loss + alpha * l2_loss
        error = y_proba - y_train_one_hot
        if iteration % 500 == 0:
            print(iteration, loss)
        gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha*Theta[1:]]
        Theta = Theta - eta * gradients

    # now check against validation data:
    logits = X_validation.dot(Theta)
    y_proba = softmax(logits)
    y_predict = np.argmax(y_proba, axis=1)
    accuracy_score = np.mean(y_predict == y_validation)
    print(accuracy_score)

    # train again, but add early stop:
    eta = 0.1
    n_iterations = 5001
    m = len(X_train)
    epsilon = 1e-7
    alpha = 0.1
    best_loss = np.infty

    Theta = np.random.randn(n_inputs, n_outputs)

    for iteration in range(n_iterations):
        logits = X_train.dot(Theta)
        y_proba = softmax(logits)
        xentropy_loss = -np.mean(np.sum(y_train_one_hot * np.log(y_proba + epsilon), axis=1))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
        loss = xentropy_loss + alpha * l2_loss
        error = y_proba - y_train_one_hot
        gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
        Theta = Theta - eta * gradients

        logits = X_validation.dot(Theta)
        y_proba = softmax(logits)
        xentropy_loss = -np.mean(np.sum(y_validation_one_hot * np.log(y_proba + epsilon), axis=1))
        l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
        loss = xentropy_loss + alpha * l2_loss
        if iteration % 500 == 0:
            print(iteration, loss)
        if loss < best_loss:
            best_loss = loss
        else:
            print(iteration - 1, best_loss)
            print(iteration, loss, "early stopping")
            break

    # now evaluate:
    logits = X_validation.dot(Theta)
    y_proba = softmax(logits)
    y_predict = np.argmax(y_proba, axis=1)
    accuracy_score = np.mean(y_predict == y_validation)
    print(accuracy_score)

    x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )

    X_new = np.c_[x0.ravel(), x1.ravel()]
    X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

    logits = X_new_with_bias.dot(Theta)
    y_proba = softmax(logits)
    y_predict = np.argmax(y_proba, axis=1)

    zz1 = y_proba[:, 1].reshape(x0.shape)
    zz = y_predict.reshape(x0.shape)

    plt.figure(figsize=(10, 5))
    plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris virginica")
    plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris versicolor")
    plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris setosa")

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

    plt.contourf(x0, x1, zz, cmap=custom_cmap)
    contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
    plt.clabel(contour, inline=1, fontsize=12)
    plt.xlabel("Petal length", fontsize=14)
    plt.ylabel("Petal width", fontsize=14)
    plt.legend(loc="upper left", fontsize=14)
    plt.show()
