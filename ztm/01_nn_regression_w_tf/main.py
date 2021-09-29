# introduction to regression with neural networks in tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(tf.__version__)
    X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])

    y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

    plt.scatter(X, y, c="red")
    plt.show()

    # examining desired output shape
    house_info = tf.constant(["bedroom", "bathroom", "garage"])
    house_price = tf.constant([939700])
    print(house_info, house_price)

    input_shape = X[0].shape
    output_shape = y[0].shape
    print(input_shape, output_shape) # scalar values, so output has no shape
    print(X[0], y[0])

    # turn numpy arrays into tensors and check shapes again
    X = tf.cast(tf.constant(X), tf.float32)
    y = tf.cast(tf.constant(y), tf.float32)
    print(X, y)
    input_shape = X[0].shape
    output_shape = y[0].shape
    print(input_shape, output_shape)  # still no dims to shape

    # Steps to modelling with tensorflow
    # Prep data - format it to tensors, clean it up
    # Create model, input/output layers, hidden layers if needed
    # Compile model - define loss function, optimizer (how the model will improve itself) and evaluation metric (how to interpret its performance)
    # Fit model
    # Evaluate
    # Improve through experimentation
    # Save and reload if needed

    tf.random.set_seed(42)
    # Create model using Sequential API
    # Sequential groups a linear stack of layers for model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    # Compile model
    # Mean Absolute Error (MAE): loss = mean(abs(y_true - y_pred), axis=1)
    # Stochastic Gradient Descent (SGD): tells NN how to improve
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])

    # Fit model
    model.fit(X, y, epochs=5)

    # try prediction
    print(X, y)
    print(model.predict([17.]))

    # Improving model
    # Creating - more layers, more hidden units, change activation function
    # Compiling - change optimization function, learning rate,
    # Fitting - more epochs, more data


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
