# simple starting model
import numpy as np
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Steps to build NN model
# 1. Prepare the data
# 2. Build the model: inputs, outputs, layers
# 3. Compile the model: loss function, optimizer, metrics
# 4. Fit the model to the training data
# 5. Evaluate and improve through experimentation


"""
Function to help visualize predictions;
params: trained model and X,y predictions
create meshgrid of X values
make predictions across meshgrid
plot predictions along with lines along decision zone boundaries
"""
def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by model predicting on X
    :param model:
    :param X:
    :param y:
    :return:
    """
    #define axis boundaries of the plot
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1  # add/subtract 0.1 to add a bit of margin
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    # create meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    # create X values to use tro make predictions
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2d arrays
    # print(xx.shape, yy.shape)
    # print(xx, yy)
    # print(x_in.shape)

    # make predictions
    y_pred = model.predict(x_in)

    # check for multi-class
    if len(y_pred[0]) > 1:
        print("this is multiclass classification")
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("this is binary classifiication")
        y_pred = np.round(y_pred).reshape(xx.shape)

    # plot the decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.show()


def run(X, y):
    """
    tf.random.set_seed(42)
    model = tf.keras.Sequential(
        tf.keras.layers.Dense(1)
    )
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["accuracy"])
    model.fit(X, y, epochs=5)

    model.fit(X, y, epochs=200, verbose=0)
    print(model.evaluate(X, y))
    """

    """
    tf.random.set_seed(42)
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1),
        tf.keras.layers.Dense(1),
    ])

    model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.SGD(), metrics=["accuracy"])
    model_1.fit(X, y, epochs=100, verbose=0, workers=-1)
    print(model_1.evaluate(X, y))
    """


    """
    Common ways to improve model performance:
        Adding layers
        Increase the number of hidden units in the layers
        Change the activation functions of the layers
        Change the optimization function of the model
        Change the learning rate of the optimization function
        Fit on more data
        Fit for longer    
    """

    tf.random.set_seed(42)
    model_2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
    ])

    model_2.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["accuracy"])
    model_2.fit(X, y, epochs=100, verbose=0, workers=-1)
    print(model_2.evaluate(X, y))

    # visualize predictions:
    plot_decision_boundary(model_2, X, y)

