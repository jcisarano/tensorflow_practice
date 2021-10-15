# multiclass classification
# NN that classifies different articles of clothing
# using tf fashion_mnist dataset
# 10 classes, 60k examples, test set 10k, 28x28 image
import matplotlib.pyplot as plt
import pandas as pd
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


def plot_random_image(model, images, true_labels, classes):
    import random
    """
    picks a random image, plots it and labels it with a prediction and truth label.
    :param model: 
    :param images: 
    :param true_labels: 
    :param classes: 
    :return: 
    """
    i = random.randint(0, len(images))
    target_image = images[i]
    pred_probs = model.predict(target_image.reshape(1, 28, 28))
    pred_label = classes[pred_probs.argmax()]
    true_label = classes[true_labels[i]]
    plt.imshow(target_image, cmap=plt.cm.binary)
    if pred_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label,
                                                     100*tf.reduce_max(pred_probs),
                                                     true_label),
               color=color)
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
    """tf.random.set_seed(42)
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
                                  workers=-1)"""
    # normalization improved accuracy to about 80%, from 35% when not normalized

    # plot loss curves
    # pd.DataFrame(norm_history.history).plot(title="Non normalized data")
    # pd.DataFrame(non_norm_history.history).plot(title="Normalized data")
    # plt.show()

    # work on finding ideal learning rate (where loss decreases the most)
    """tf.random.set_seed(42)
    model_lr = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model_lr.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))
    history_find_lr = model_lr.fit(train_data_norm,
                                   tf.one_hot(train_labels, depth=10),
                                   epochs=40,
                                   validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)),
                                   workers=-1,
                                   callbacks=[lr_scheduler])"""

    # plot learning rate decay curve
    """lrs = 1e-3 * (10 ** (tf.range(40) / 20))
    plt.semilogx(lrs, history_find_lr.history["loss"])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.title("Finding the ideal learning rate")
    plt.show()"""

    # ideal learning rate appearst to be 0.001
    tf.random.set_seed(42)
    model_ideal_lr = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model_ideal_lr.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                           optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           metrics=["accuracy"])
    history_ideal_lr = model_ideal_lr.fit(train_data_norm,
                                          tf.one_hot(train_labels, depth=10),
                                          epochs=10,
                                          validation_data=(test_data_norm, tf.one_hot(test_labels, depth=10)))

    y_pred_probabilities = model_ideal_lr.predict(test_data_norm)
    y_preds = y_pred_probabilities.argmax(axis=1)

    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y_true=test_labels, y_pred=y_preds))

    # assess outputs
    # create a confusion matrix
    from evaluation import plot_confusion_matrix
    plot_confusion_matrix(test_labels, y_preds,
                          classes=class_names, figsize=(15, 15), text_size=10)

    for i in range(10):
        plot_random_image(model=model_ideal_lr, images=test_data_norm, true_labels=test_labels, classes=class_names)

    # What patterns is the model learning?
    # Layers of most recent model. Each layer has specific role in finding data features.
    print(model_ideal_lr.layers)
    print(model_ideal_lr.layers[0])

    weights, biases = model_ideal_lr.layers[1].get_weights()
    print(weights, weights.shape)

    print(biases, biases.shape)

    # from tensorflow.keras.utils import plot_model
    # plot_model(model_ideal_lr, show_shapes=True)
