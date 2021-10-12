# multiclass classification
# NN that classifies different articles of clothing
# using tf fashion_mnist dataset
# 10 classes, 60k examples, test set 10k, 28x28 image

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


def load_data():
    # data is already sorted into training and test sets
    return fashion_mnist.load_data()

def plot_one_data_sample(example):
    import matplotlib.pyplot as plt
    plt.imshow(example)
    plt.show()


def run():
    (train_data, train_labels), (test_data, test_labels) = load_data()
    # print(f"Training sample:\n {train_data[0]}")
    # print(f"Training label:\n {train_labels[0]}")
    # plot_one_data_sample(train_data[7])
    
