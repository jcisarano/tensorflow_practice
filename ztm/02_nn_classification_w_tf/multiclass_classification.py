# multiclass classification
# NN that classifies different articles of clothing
# using tf fashion_mnist dataset
# 10 classes, 60k examples, test set 10k, 28x28 image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def load_data():
    # data is already sorted into training and test sets
    return fashion_mnist.load_data()


def plot_one_data_sample(example, name):
    import matplotlib.pyplot as plt
    plt.imshow(example, cmap=plt.cm.binary)
    plt.title(name)
    plt.show()


def plot_multiple_random_samples(t_data, t_labels, c_names):
    import random
    plt.figure(figsize=(7, 7))
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        rand_index = random.choice(range(len(t_data)))
        plt.imshow(t_data[rand_index], cmap=plt.cm.binary)
        plt.title(c_names[t_labels[rand_index]])
        plt.axis(False)
    plt.show()


def run():
    (train_data, train_labels), (test_data, test_labels) = load_data()
    # print(f"Training sample:\n {train_data[0]}")
    # print(f"Training label:\n {train_labels[0]}")
    # plot_one_data_sample(train_data[7])

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot", ]

    # index = 440
    # plot_one_data_sample(train_data[index], class_names[train_labels[index]])

    # plot multiple random images to help visualize data
    # plot_multiple_random_samples(t_data=train_data, t_labels=train_labels, c_names=class_names)

    # multiclass classifier
    # input shape: 28x28, the image size
    # output shape: 10, one per type of clothing
    # Loss function for one-hot encoded labels: tf.keras.losses.CategoricalCrossentropy
    # Loss function for integer labels: SparseCategoricalCrossentropy

    """tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # converts 28x28 image data into one long vector (None,784)
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),  # used when labels are not one-hot encoded
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])

    non_norm_history = model.fit(train_data,
                                 tf.one_hot(train_labels, depth=10),
                                 epochs=10, validation_data=(test_data, tf.one_hot(test_labels, depth=10)),
                                 workers=-1)

    # check the model summary
    print(model.summary())"""

    # check size of data
    print(train_data.min(), train_data.max())
    # neural networks work best with data normalized to 0-1 range
    train_data_norm = train_data / train_data.max()
    test_data_norm = test_data / test_data.max()
    print(train_data_norm.min(), train_data_norm.max())

    # new model that uses the normalized data
    tf.random.set_seed(42)
    model_norm = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    model_norm.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=["accuracy"])
    norm_history = model_norm.fit(train_data_norm,
                                  tf.one_hot(train_labels, depth=10),
                                  epochs=10,
                                  validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)),
                                  workers=-1)
