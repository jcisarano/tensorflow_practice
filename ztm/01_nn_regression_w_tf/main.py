# introduction to regression with neural networks in tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    # plots training, test data and predictions against ground truth
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="blue", label="Training data")
    plt.scatter(test_data, test_labels, c="green", label="Testing data")
    plt.scatter(test_data, predictions, c="red", label="Predictions")
    plt.legend()
    plt.show()


def mae(y_true, y_pred):
    return tf.metrics.mean_absolute_error(y_true=y_true, y_pred=tf.squeeze(y_pred))


def mse(y_true, y_pred):
    return tf.metrics.mean_squared_error(y_true=y_true, y_pred=tf.squeeze(y_pred))

if __name__ == '__main__':
    print(tf.__version__)
    """
    X = np.array([-7., -4., -1., 2., 5., 8., 11., 14.])

    y = np.array([3., 6., 9., 12., 15., 18., 21., 24.])

    # plt.scatter(X, y, c="red")
    # plt.show()

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
    """

    # Steps to modelling with tensorflow
    # Prep data - format it to tensors, clean it up
    # Create model, input/output layers, hidden layers if needed
    # Compile model - define loss function, optimizer (how the model will improve itself) and evaluation metric (how to interpret its performance)
    # Fit model
    # Evaluate
    # Improve through experimentation
    # Save and reload if needed

    """
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
    """

    """
    # Improving model
    # Creating - more layers, more hidden units, change activation function
    # Compiling - change optimization function, learning rate,
    # Fitting - more epochs, more data

    # redo the model, but with more training epochs
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.SGD(),
                  metrics=["mae"])

    model.fit(X, y, epochs=100)

    print(X, y)
    print(model.predict([17.]))


    # try other tweaks
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="None"),
        tf.keras.layers.Dense(1)
    ])

    # the learning rate can be the most important hyper parameter to tweak
    model.compile(loss=tf.keras.losses.mae,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                  metrics=["mae"])
    model.fit(X, y, epochs=100)
    print(X, y)
    print(model.predict([17., 20., 35., 88., 150.]))
    """

    # typical workflow
    # build model -> fit -> evaluate -> tweak -> fit -> evaluate -> tweak ...

    # evaluation means "visualize, visualize, visualize"
    # what data are we working with? what does it look like?
    # what does the model look like?
    # how does the model perform while it learns?
    # how does the predictions line up against the ground truth (the original labels)?

    tf.random.set_seed(42)
    # a bigger test case, with a bigger dataset
    # creates tensor with values from -100 to 100 in steps of 4
    X = tf.range(-100, 100, 4)
    # X = tf.random.shuffle(X, seed=42)  # shuffled data is better for training, I think? Avoids patterns due to order

    # labels (same relationship as before: X val plus 10)
    y = X + 10

    # plt.scatter(X, y)
    # plt.show()

    # test-train sets
    # training set is what the model learns from, usually 70-80% of your data
    # validation set - tunes the model, aka dev set, 10-15% of total data
    # test set used to evaluate the model, 10-15% of total data

    # split the data into training and test sets, 80/20 split
    split_index = int(len(X) * 0.8)
    X_train = X[:split_index]  # slices array up to the index (exclusive)
    X_test = X[split_index:]  # slices array from index to the end (inclusive)
    y_train = y[:split_index]
    y_test = y[split_index:]
    print(X_train, X_test)
    print(y_train, y_test)

    # Build neural network for data:
    # create model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=[1], name="output_layer")  # input shape is shape of desired output, in this case a single scalar
    ], name="test_model")

    model.compile(
        loss=tf.keras.losses.mae,
        optimizer=tf.keras.optimizers.SGD(),
        metrics=["mae"]
    )

    # shows info on model, e.g. layers, output shape, # params
    # Dense layer == fully connected layers, i.e. every node in layer A connects to every other node in layer B
    print(model.summary())

    # from tensorflow.keras.utils import plot_model
    # plot_model(model=model, show_shapes=True)

    # model.fit(X_train, y_train, epochs=100, verbose=0)

    # plot predictions against ground truth labels
    # e.g. y_test vs y_pred
    y_pred = model.predict(X_test)

    plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test, test_labels=y_test, predictions=y_pred)

    # main evaluation metrics for regression are MAE and MSE
    print(model.evaluate(X_test, y_test))

    # another way to calculate MAE
    # need to squeeze y_pred because its shape is (10,1) where y_text is (10,)
    ma = mae(y_test, y_pred)
    print(ma)

    # calculate MSE
    ms = mse(y_test, y_pred)
    print(ms)

    # experiments to improve the model
    # build -> fit -> evaluate -> tweak ->fit ....
    # get more data
    # increase model size, more layers, more hidden units per layer,
    # train longer

    # same as original, trained 100 epochs
    model_1 = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model_1.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["mae"])
    model_1.fit(X_train, y_train, epochs=100, verbose=0)
    y_pred_1 = model_1.predict(X_test)
    plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test,
                     test_labels=y_test, predictions=y_pred_1)

    mae_1 = mae(y_true=y_test, y_pred=tf.squeeze(y_pred_1))
    mse_1 = mse(y_test, y_pred_1)
    print(mae_1, mse_1)

    # two layers, trained 100 epochs
    model_2 = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model_2.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["mae"])
    model_2.fit(X_train, y_train, epochs=100, verbose=0)
    y_pred_2 = model_2.predict(X_test)
    plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test,
                     test_labels=y_test, predictions=y_pred_2)
    mae_2 = mae(y_test, y_pred_2)
    mse_2 = mse(y_test, y_pred_2)
    print(mae_2, mse_2)

    # two layers, trained 500 epochs
    model_3 = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1)
    ])
    model_3.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=["mae"])
    model_3.fit(X_train, y_train, epochs=500, verbose=0)
    y_pred_3 = model_3.predict(X_test)
    plot_predictions(train_data=X_train, train_labels=y_train, test_data=X_test,
                     test_labels=y_test, predictions=y_pred_3)
    mae_3 = mae(y_test, y_pred_3)
    mse_3 = mae(y_test, y_pred_3)
    print(mae_3, mse_3)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
