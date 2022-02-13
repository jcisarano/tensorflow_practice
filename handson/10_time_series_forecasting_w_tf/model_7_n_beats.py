"""
Implementation of n-beats algorithm as described in https://arxiv.org/pdf/1905.10437.pdf

From page 3 fig 1:
'Proposed architecture. The basic building block is a multi-layer FC network with RELU
nonlinearities. It predicts basis expansion coefficients both forward, θf, (forecast) and backward, θb,
(backcast). Blocks are organized into stacks using doubly residual stacking principle. A stack may have layers with
shared gb and gf. Forecasts are aggregated in hierarchical fashion. This enables
building a very deep neural network with interpretable outputs.'

 Blocks > Stacks > Deep Network Model

Uses TensorFlow layer subclassing to make custom layers and also Functional API for its custom architecture

N-BEATS steps
    # Set up instance of N-BEATS block layer using NBeatsBlock class - this will be the initial block, the rest are
        created by stacks
    # Create an input layer for the N_BEATS stack (using Keras functional API)
    # Make the initial backcast and forecast for the model with block layer created above
    # Use for loop to create stacks of block layers
    # Use the NBeatsBlock class within for loop to create blocks which return backcasts and block-level forecasts
    # Create the doubly residual stacking using subtract and add layers
    # Put the model inputs and outputs together using tf.keras.Model()
    # Compile with MAE loss (however, the paper uses multiple losses) and Adam() optimizer
    # Fit N-BEATS for 5000 epochs using these callbacks:
        - Early stopping - Stop training if the model stops improving
        - Reduce learning rate on plateau - Lower LR if the model stops improving, taking smaller steps to improve
"""
import tensorflow as tf

from utils import load_data, load_dataframe

WINDOW_SIZE: int = 7
HORIZON: int = 1
BATCH_SIZE: int = 1024
N_EPOCHS: int = 5000
N_NEURONS: int = 512
N_LAYERS: int = 4
N_STACKS: int = 30

INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON


class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                 input_size: int, theta_size: int,
                 horizon: int, n_neurons: int,
                 n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.horizon = horizon
        self.input_size = input_size  # same as window size in previous models
        self.theta_size = theta_size
        self.n_neurons = n_neurons
        self.n_layers = n_layers

        # block contains stack of 4 fully connected layers with ReLU activation
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]

        # Output of block is theta layer with linear activation
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)

        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]

        return backcast, forecast


def test_nbeats_block_class():
    tf.random.set_seed(42)
    # test NBeatsBlock class
    dummy_nbleats_block_layer = NBeatsBlock(input_size=WINDOW_SIZE,
                                            theta_size=WINDOW_SIZE + HORIZON,  # backcast + forecast
                                            horizon=HORIZON,
                                            n_neurons=128,
                                            n_layers=4)

    # create dummy inputs (important that they have the same size as input_size)
    dummy_inputs = tf.expand_dims(tf.range(WINDOW_SIZE) + 1, axis=0)  # input shape to model must reflect input shape
    print(dummy_inputs)
    backcast, forecast = dummy_nbleats_block_layer(dummy_inputs)
    # these are randomized activation outputs of theta layer (no training has happened yet)
    print(f"Backcast: {tf.squeeze(backcast.numpy())}")
    print(f"Forecast: {tf.squeeze(forecast.numpy())}")


def make_datasets():
    prices = load_dataframe()
    prices_nbeats = prices.copy()
    for i in range(WINDOW_SIZE):
        prices_nbeats[f"Price+{i + 1}"] = prices_nbeats["Price"].shift(periods=i + 1)
    # print(prices_nbeats.head())

    # make features and labels
    X = prices_nbeats.dropna().drop("Price", axis=1)
    y = prices_nbeats.dropna()["Price"]

    # make train and test sets
    split_size = int(len(X) * 0.8)
    X_train, y_train = X[:split_size], y[:split_size]
    X_test, y_test = X[split_size:], y[split_size:]
    # print(len(X_train), len(y_train), len(X_test), len(y_test))

    return X_train, X_test, y_train, y_test


def batch_and_prefetch_datasets(X_train, X_test, y_train, y_test):
    # using tf.data API will make dataset more performant
    # this is more useful for very large datasets
    train_features_dataset = tf.data.Dataset.from_tensor_slices(X_train)
    train_labels_dataset = tf.data.Dataset.from_tensor_slices(y_train)

    test_features_dataset = tf.data.Dataset.from_tensor_slices(X_test)
    test_labels_dataset = tf.data.Dataset.from_tensor_slices(y_test)

    # combine labels and features to tuple (features, labels)
    train_dataset = tf.data.Dataset.zip((train_features_dataset, train_labels_dataset))
    test_dataset = tf.data.Dataset.zip((test_features_dataset, test_labels_dataset))

    # batch and prefetch
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE)  # autotune determines # of available CPUs
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def tf_math_examples():
    # n-beats uses doubly residual stacking (aka skip connections) to help train deeper architecture (see section 3.2)
    # examples of subtract and add layers which will be used for residual connections later
    tensor_1 = tf.range(10) + 10
    tensor_2 = tf.range(10)

    subtracted = tf.keras.layers.subtract([tensor_1, tensor_2])
    added = tf.keras.layers.add([tensor_1, tensor_2])

    print(f"Input tensors: {tensor_1.numpy()} & {tensor_2.numpy()} ")
    print(f"Subtracted: {subtracted.numpy()}")
    print(f"Added: {added.numpy()}")


def run():
    # test_nbeats_block_class()

    X_train, X_test, y_train, y_test = make_datasets()
    train_dataset, test_dataset = batch_and_prefetch_datasets(X_train, X_test, y_train, y_test)
    # print(train_dataset, test_dataset)

    tf.random.set_seed(42)

    # Create initial NBeatsBlock
    nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                     theta_size=THETA_SIZE,
                                     horizon=HORIZON,
                                     n_neurons=N_NEURONS,
                                     n_layers=N_LAYERS,
                                     name="InitalBlock")

    # Create input layer for stack
    stack_input = tf.keras.layers.Input(shape=(INPUT_SIZE), name="StackInput")

    # Create initial backcast and forecast
    residuals, forecast = nbeats_block_layer(stack_input)

    # Create stacks of block layers
    for i, _ in enumerate(range(N_STACKS - 1)):  # subtract 1 because 1st stack is created above

        # Use NBeatsBlock to calc backcast and forecast:
        backcast, block_forecast = NBeatsBlock(
            input_size=INPUT_SIZE,
            theta_size=THETA_SIZE,
            horizon=HORIZON,
            n_neurons=N_NEURONS,
            n_layers=N_LAYERS,
            name=f"NBeatsBlock_{i}"
        )(residuals)  # pass in the residuals

        # Create doubly residual stacking
        residuals = tf.keras.layers.subtract([residuals, backcast], name=f"Subtract_{i}")
        forecast = tf.keras.layers.add([forecast, block_forecast], name=f"Add_{i}")

    model = tf.keras.Model(inputs=stack_input, outputs=forecast, name="Model_7_NBEATS")
    model.compile(loss="MAE", optimizer=tf.keras.optimizers.Adam())

    # Fit with early stopping and reduce lr on plateau
    history = model.fit(train_dataset,
                        epochs=N_EPOCHS,
                        validation_data=test_dataset,
                        callbacks=[
                            tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                patience=200,
                                restore_best_weights=True
                            ),
                            tf.keras.callbacks.ReduceLROnPlateau(
                                monitor="val_loss",
                                patience=100,
                                verbose=1
                            )],
                        workers=-1)

    return 0

    print("n-beats")
