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
        ax = plt.subplot(2, 2, i+1)
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

    index = 440
    plot_one_data_sample(train_data[index], class_names[train_labels[index]])

    # plot multiple random images to help visualize data
    plot_multiple_random_samples(t_data=train_data, t_labels=train_labels, c_names=class_names)
