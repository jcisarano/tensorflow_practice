# introduction to regression with neural networks in tensorflow

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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
    # print(X_train, X_test)
    # print(y_train, y_test)
    """

    """
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
    """


    # experiments to improve the model
    # build -> fit -> evaluate -> tweak ->fit ....
    # get more data
    # increase model size, more layers, more hidden units per layer,
    # train longer

    """
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
    """

    """
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
    """

    """
    # comparing results from different models using pandas DataFrame
    model_results = [["model_1", mae_1.numpy(), mse_1.numpy()],
                     ["model_2", mae_2.numpy(), mse_2.numpy()],
                     ["model_3", mae_3.numpy(), mse_3.numpy()]]
    all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
    print(all_results)
    """

    # experiments should start small and increase in size/complexity only when needed
    # try to keep time between experiments small
    # the more experiments you do, the more you will figure out what works/does not work
    # experiment, experiment, experiment

    # TensorBoard is a tf library to help track modeling experiments
    # Weights & Biases - a tool for tracking all kinds of ML experiments, works with TensorBoard

    """
    # save model to use again later
    # two formats for save: SavedModel and HDF5, determined by file extension. SavedModel is default
    # hdf5 is good for large models in general format usable by different applications outside tensorflow
    # SavedModel is good for tensorflow only use
    model_3.save(filepath='models/model_3')
    model_3.save(filepath='models/model_3_hdf5_format.h5')

    # load saved SavedModel format
    loaded_via_sm = tf.keras.models.load_model("models/model_3")
    print(loaded_via_sm.summary())
    y_pred_loaded_via_sm = loaded_via_sm.predict(X_test)
    print(y_pred_3 == y_pred_loaded_via_sm)

    # load saved HDF5 format
    loaded_via_H5 = tf.keras.models.load_model("models/model_3_hdf5_format.h5")
    print(loaded_via_H5.summary())
    y_pred_loaded_via_h5 = loaded_via_H5.predict(X_test)
    print(y_pred_3 == y_pred_loaded_via_h5)
    """


    # larger example - insurance dataset
    # to predict insurance charges based on other indicators
    # insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
    insurance = pd.read_csv("datasets/insurance.csv")
    # print(insurance)

    # need to reformat data to convert non-numerical input features into numerical features
    # we will use one-hot encoding
    insurance_one_hot = pd.get_dummies(insurance)

    y = insurance_one_hot["charges"]
    X = insurance_one_hot.drop(columns=['charges'], axis=1)
    print(X.head())
    print(y.head())

    # split_index = int(len(X) * 0.8)
    # X_train = X[:split_index]
    # y_train = y[:split_index]
    # X_test = X[split_index:]
    # y_test = y[split_index:]

    # easier way to split, does random shuffle of data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # print(len(X), len(X_train), len(X_test))


    """
    # first model
    tf.random.set_seed(42)
    insurance_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(1)
    ])

    insurance_model.compile(loss=tf.keras.losses.mae,
                            optimizer=tf.keras.optimizers.SGD(),
                            metrics=["mae"])

    insurance_model.fit(X_train, y_train, epochs=100, workers=-1, verbose=0)
    print(insurance_model.evaluate(X_test, y_test, verbose=0))

    # error is significant, given average data:
    print(y_train.median(), y_train.mean())

    # insurance_pred = insurance_model.predict(X_test, workers=-1, verbose=3)
    """

    """
    # experiments to improve model performance
    # 1 add an extra layer with more hidden units, change optimizer to Adam()
    # 2 train for longer
    # 3 ???

    tf.random.set_seed(42)
    insurance_model_2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    insurance_model_2.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["mae"], loss=tf.keras.losses.mae)
    insurance_model_2.fit(X_train, y_train, epochs=100, workers=-1, verbose=0)
    print(insurance_model_2.evaluate(X_test, y_test, verbose=0))


    tf.random.set_seed(42)
    insurance_model_3 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
    ])
    insurance_model_3.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
    history = insurance_model_3.fit(X_train, y_train, epochs=200, workers=-1, verbose=0)
    print(insurance_model_3.evaluate(X_test, y_test, verbose=0))
    """


    # pd.DataFrame(history.history).plot()
    # plt.ylabel("loss")
    # plt.xlabel("epochs")
    # plt.show()

    # improving data preprocessing: normalization and standardization
    # usual preprocessing steps:
    # 1 turn all data into numbers (nn can't use strings)
    # 2 make sure all tensor shapes fit
    # 3 scale features using normalization/standardization (nn tend to prefer normalization)

    # X["age"].plot(kind="hist")
    # X["bmi"].plot(kind="hist")
    # plt.show()

    # scikit-learn scalers
    # MinMaxScaler - normalizes all values (0-1) while maintaining the original distribution
    # StandardScaler - removes mean and divides all values by standard deviation, note: reduces the effect of outliers
    # StandardScaler creates a normal distribution, bell-shaped, Gaussian
    # Generally, it is worth trying both to see which is better

    """
    # preprocessing practice
    from sklearn.compose import make_column_transformer
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split

    tf.random.set_seed(42)
    # reimport the data to start over fresh
    insurance = pd.read_csv("datasets/insurance.csv")
    # print(insurance)
    ct = make_column_transformer(
        (MinMaxScaler(), ["age", "bmi", "children"]),
        (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
    )

    # create features and labels
    X = insurance.drop("charges", axis=1)
    y = insurance["charges"]

    # create train/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # fit column transformer to training data only
    ct.fit(X_train)

    # transform training and test data with normalization and one-hot encoding
    X_train_normal = ct.transform(X_train)
    X_test_normal = ct.transform(X_test)

    # print(X_train.loc[0])
    # print(X_train_normal[0])
    # print(X_train.shape, X_train_normal.shape)

    insurance_model_4 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1),
    ])
    insurance_model_4.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(), metrics=["mae"])
    insurance_model_4.fit(X_train_normal, y_train, workers=-1, epochs=100, verbose=0)
    print(insurance_model_4.evaluate(X_test_normal, y_test, workers=-1, verbose=0))

    # another insurance model to experiment with
    insurance_model_5 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dense(50),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])

    insurance_model_5.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["mae"])
    insurance_model_5.fit(X_train_normal, y_train, epochs=500, workers=-1)
    print(insurance_model_5.evaluate(X_test_normal, y_test, workers=-1))
    """

    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(path="boston_housing.npz", test_split=0.2, seed=42)

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    boston_model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1),
    ])
    boston_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["mae"])
    boston_model.fit(X_train, y_train, epochs=200, workers=-1, verbose=0)
    # print(boston_model.predict(X_test))
    print(boston_model.evaluate(X_test, y_test, workers=-1, verbose=0))


    # try boston data with scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    boston_model_2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    boston_model_2.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=["mae"])
    boston_model_2.fit(X_train_normalized, y_train, epochs=200, workers=-1, verbose=0)
    print(boston_model_2.evaluate(X_test_normalized, y_test, workers=-1, verbose=0))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
